use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::path::Path;
use std::process::{exit, Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::future::join_all;
use qdrant_client::prelude::*;
use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
use qdrant_client::qdrant::vectors::VectorsOptions;
use qdrant_client::qdrant::{PointId, PointsIdsList, PointsSelector, Vectors};
use tokio::time::sleep;

const QDRANT_BIN: &str = "/home/timvisee/git/qdrant/target/perf/qdrant";
const BFB_BIN: &str = "bfb";
const BFB_PARAMS: &str = "--segments 2 --on-disk-vectors true --on-disk-hnsw --on-disk-payload";
// const BFB_PARAMS: &str = "--segments 2";
// const BFB_PARAMS: &str = "";
const HOST: &str = "http://localhost:6334";
const COLLECTION: &str = "test";
const COLLECTIONS_DIR: &str = "./storage/collections";
const N: usize = 100_000;
const INDEX_THRESHOLD: usize = N / 100;
const BATCH_SIZE: usize = 100;
const JOB_SECONDS: u64 = 7;
const JOB_COUNT: usize = 20;
const USE_BACKUP: bool = true;
const BACKUP_COLLECTION: &str = "test-backup";
const QUIT_ON_FINISH: bool = true;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Arc::new(QdrantClient::from_url(HOST).build()?);

    prepare(&client).await;
    setup(client.clone()).await;
    let ok = test(&client).await;

    if ok {
        cleanup();
    } else {
        exit(1);
    }

    Ok(())
}

async fn prepare(_client: &QdrantClient) {
    let qdrant_bin = Path::new(QDRANT_BIN).file_name().unwrap().to_str().unwrap();

    // Kill running Qdrant and bfb
    system(format!("pkill -x {qdrant_bin}"));
    system(format!("pkill -x {BFB_BIN}"));
    system(format!("pidwait -x {qdrant_bin}"));

    // Wipe test collection from disk
    system(format!("rm -r {COLLECTIONS_DIR}/{COLLECTION}"));
}

async fn use_backup(client: &QdrantClient) -> Child {
    println!("Using backed-up collection");

    system(format!(
        "cp -r {COLLECTIONS_DIR}/{BACKUP_COLLECTION} {COLLECTIONS_DIR}/{COLLECTION}"
    ));

    // Start Qdrant
    let qdrant_handle = spawn(format!(
        "QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=0 {QDRANT_BIN}"
    ));
    wait_qdrant_ready(client).await;

    qdrant_handle
}

async fn build_new(client: &QdrantClient) -> Child {
    // Start Qdrant
    let mut qdrant_handle = spawn(format!(
        "QDRANT__STORAGE__PERFORMANCE__MAX_OPTIMIZATION_THREADS=0 {QDRANT_BIN}"
    ));
    wait_qdrant_ready(client).await;

    // Insert base data
    system(format!(
        "{BFB_BIN} --collection-name={COLLECTION} -n={N} --indexing-threshold={INDEX_THRESHOLD} {BFB_PARAMS}",
    ));

    if USE_BACKUP {
        let qdrant_bin = Path::new(QDRANT_BIN).file_name().unwrap().to_str().unwrap();
        system(format!("pkill -x {qdrant_bin}"));
        system(format!("pidwait -x {qdrant_bin}"));
        system(format!(
            "cp -r {COLLECTIONS_DIR}/{COLLECTION} {COLLECTIONS_DIR}/{BACKUP_COLLECTION}"
        ));

        // Start Qdrant
        qdrant_handle = spawn(QDRANT_BIN);
        wait_qdrant_ready(client).await;
    }

    qdrant_handle
}

/// Set up potentially broken collection.
async fn setup(client: Arc<QdrantClient>) {
    let backup_path = Path::new(COLLECTIONS_DIR).join(BACKUP_COLLECTION);
    let mut qdrant_handle = if USE_BACKUP && backup_path.is_dir() {
        use_backup(&client).await
    } else {
        build_new(&client).await
    };

    // // Start Qdrant
    // let mut qdrant_handle = spawn(QDRANT_BIN);
    // wait_qdrant_ready(&client).await;

    // // Insert base data
    // system(format!(
    //     "{BFB_BIN} --collection-name={COLLECTION} -n={N} --indexing-threshold={INDEX_THRESHOLD} {BFB_PARAMS}",
    // ));

    // Spawn jobs to set random payloads
    let jobs = join_all(
        (0..JOB_COUNT)
            .map({
                let client = client.clone();
                move |_| set_random_payload(client.clone())
            })
            .collect::<Vec<_>>(),
    );
    let jobs_handle = tokio::task::spawn(jobs);

    sleep(Duration::from_secs(JOB_SECONDS)).await;

    // test(&client).await;

    // SIGKILL Qdrant, stop jobs
    qdrant_handle.kill().expect("failed to kill qdrant process");
    jobs_handle.abort();

    sleep(Duration::from_secs(1)).await;

    // Restart Qdrant
    let _qdrant_handle = spawn(QDRANT_BIN);
    wait_qdrant_ready(&client).await;
}

/// Test if the collection we end up with is sound.
///
/// - checks if all points exist
/// - checks if all points contain vector data
/// - checks if all points contain vector payload
async fn test(client: &QdrantClient) -> bool {
    // Test all points
    let (mut no_points, mut no_vectors, mut no_payload) = (vec![], vec![], vec![]);
    for batch_id in (0..N).step_by(BATCH_SIZE) {
        // Build list of batch point IDs, request point data
        let ids = batch_id..batch_id + BATCH_SIZE;
        let point_ids = ids
            .clone()
            .map(|id| PointId::from(id as u64))
            .collect::<Vec<_>>();
        let points = client
            .get_points(COLLECTION, &point_ids, Some(true), Some(true), None)
            .await
            .unwrap()
            .result;

        // Record missing points, vectors and payload
        for id in ids {
            let point = points
                .iter()
                .find(|point| point.id == Some(PointId::from(id as u64)));
            if point.is_none() {
                no_points.push(id);
            }
            if !point
                .and_then(|point| point.vectors.clone())
                .map(has_vector_data)
                .unwrap_or(false)
            {
                no_vectors.push(id);
            }
            if point.map(|point| point.payload.is_empty()).unwrap_or(true) {
                no_payload.push(id);
            }
        }
    }

    let point_ranges = numbers_to_consecutive_ranges(no_points.clone());
    let vector_ranges = numbers_to_consecutive_ranges(no_vectors.clone());
    let payload_ranges = numbers_to_consecutive_ranges(no_payload.clone());

    let ok = no_points.is_empty() && no_vectors.is_empty() && payload_ranges.len() <= 1;

    println!("\n========================================\n");

    println!("Total points: {N}");

    println!("\nMissing points: {}", no_points.len());
    for range in point_ranges {
        println!("- {:?}: {} points", &range, range.clone().count());
    }

    println!("\nMissing vectors: {}", no_vectors.len());
    for range in vector_ranges {
        println!("- {:?}: {} vectors", &range, range.clone().count());
    }

    println!(
        "\nMissing payloads: {} (maybe not set in test)",
        no_payload.len(),
    );
    for range in payload_ranges {
        println!("- {:?}: {} payloads", &range, range.clone().count());
    }

    ok
}

fn cleanup() {
    if QUIT_ON_FINISH {
        println!("\n========================================\n");

        let qdrant_bin = Path::new(QDRANT_BIN).file_name().unwrap().to_str().unwrap();

        // Kill running Qdrant
        system(format!("pkill -x {qdrant_bin}"));
        system(format!("pidwait -x {qdrant_bin}"));
    }
}

async fn wait_qdrant_ready(client: &QdrantClient) {
    while client.list_collections().await.is_err() {
        sleep(Duration::from_secs(1)).await;
    }
}

fn system(cmd: impl AsRef<str>) {
    println!("+ {}", cmd.as_ref());
    Command::new("bash")
        .args(["-c", cmd.as_ref()])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .expect("failed to execute system command");
}

fn spawn(cmd: impl AsRef<str>) -> Child {
    println!("+ {} &", cmd.as_ref());
    Command::new("bash")
        .args(["-c", cmd.as_ref()])
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("failed to execute system command")
}

async fn set_random_payload(client: Arc<QdrantClient>) {
    loop {
        for batch_id in (0..N).step_by(BATCH_SIZE) {
            let ids = batch_id..batch_id + BATCH_SIZE;
            let point_ids = ids
                .clone()
                .map(|id| PointId::from(id as u64))
                .collect::<Vec<_>>();
            let points = PointsSelector {
                points_selector_one_of: Some(PointsSelectorOneOf::Points(PointsIdsList {
                    ids: point_ids,
                })),
            };

            // let payload = HashMap::from([("dummy", "content".into())]).into();
            let payload = HashMap::new().into();
            let result = client.set_payload(COLLECTION, &points, payload, None).await;

            // let result = client.clear_payload(COLLECTION, Some(points), None).await;

            // let payload = HashMap::new().into();
            // let result = client.update_vectors(COLLECTION, &points, payload, None).await;

            if let Err(err) = result {
                eprintln!("Set random payload error: {err}");
            }
        }
    }
}

/// Check if vectors object contains any vector data.
fn has_vector_data(vectors: Vectors) -> bool {
    match vectors.vectors_options.as_ref().unwrap() {
        VectorsOptions::Vector(vector) => {
            assert!(!vector.data.is_empty());
            true
        }
        VectorsOptions::Vectors(vectors) => {
            if vectors.vectors.is_empty() {
                return false;
            }

            vectors
                .vectors
                .iter()
                .all(|vector| !vector.1.data.is_empty())
        }
    }
}

fn numbers_to_consecutive_ranges(mut numbers: Vec<usize>) -> Vec<RangeInclusive<usize>> {
    numbers.sort_unstable();
    numbers
        .into_iter()
        .fold(vec![], |mut ranges, id| -> Vec<RangeInclusive<usize>> {
            match ranges.last_mut().filter(|range| *range.end() == id - 1) {
                Some(last) => *last = *last.start()..=id,
                None => ranges.push(id..=id),
            }
            ranges
        })
}
