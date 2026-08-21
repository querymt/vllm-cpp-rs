use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};

use hf_hub::api::sync::ApiBuilder;
use hf_hub::api::RepoInfo;
use hf_hub::{Cache, Repo, RepoType};
use serde_json::Value;

use crate::HuggingFaceError;

const CONFIG: &str = "config.json";
const TOKENIZER: &str = "tokenizer.json";
const TOKENIZER_CONFIG: &str = "tokenizer_config.json";
const SAFETENSORS: &str = "model.safetensors";
const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
const DEFAULT_REVISION: &str = "main";

/// A synchronous Hugging Face model resolver.
///
/// Resolution is separate from [`crate::Engine::load`]. The resolver returns a
/// standalone GGUF path or a sparse, runtime-complete Safetensors snapshot
/// directory in the normal Hugging Face cache layout.
#[derive(Clone)]
pub struct HuggingFaceModel {
    repo_id: String,
    revision: String,
    kind: ModelKind,
    cache_dir: Option<PathBuf>,
    token: Option<String>,
    progress: bool,
    offline: bool,
}

#[derive(Clone, Debug)]
enum ModelKind {
    Gguf { filename: String },
    Safetensors,
}

impl fmt::Debug for HuggingFaceModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HuggingFaceModel")
            .field("repo_id", &self.repo_id)
            .field("revision", &self.revision)
            .field("kind", &self.kind)
            .field("cache_dir", &self.cache_dir)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("progress", &self.progress)
            .field("offline", &self.offline)
            .finish()
    }
}

impl HuggingFaceModel {
    /// Selects one standalone GGUF file from the repository's `main` revision.
    #[must_use]
    pub fn gguf(repo_id: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: DEFAULT_REVISION.to_owned(),
            kind: ModelKind::Gguf {
                filename: filename.into(),
            },
            cache_dir: None,
            token: None,
            progress: false,
            offline: false,
        }
    }

    /// Selects a runtime-complete Safetensors directory from the repository's `main` revision.
    #[must_use]
    pub fn safetensors(repo_id: impl Into<String>) -> Self {
        Self {
            repo_id: repo_id.into(),
            revision: DEFAULT_REVISION.to_owned(),
            kind: ModelKind::Safetensors,
            cache_dir: None,
            token: None,
            progress: false,
            offline: false,
        }
    }

    /// Overrides the repository revision with a branch, tag, or commit.
    #[must_use]
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = revision.into();
        self
    }

    /// Overrides the Hugging Face Hub cache directory.
    ///
    /// The path is the Hub cache itself (for example, `~/.cache/huggingface/hub`),
    /// not its parent.
    #[must_use]
    pub fn cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    /// Overrides the cached Hugging Face token for this resolver.
    #[must_use]
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Enables or disables download progress bars. Progress is disabled by default.
    #[must_use]
    pub fn progress(mut self, progress: bool) -> Self {
        self.progress = progress;
        self
    }

    /// Enables or disables cache-only resolution. Offline mode never builds an API.
    #[must_use]
    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Resolves the selected model into the normal Hugging Face cache.
    pub fn resolve(&self) -> Result<PathBuf, HuggingFaceError> {
        self.validate()?;
        let cache = self.cache();
        if self.offline {
            return self.resolve_offline(&cache);
        }

        match &self.kind {
            ModelKind::Gguf { filename } => self.resolve_gguf_online(cache, filename),
            ModelKind::Safetensors => self.resolve_safetensors_online(cache),
        }
    }

    fn validate(&self) -> Result<(), HuggingFaceError> {
        validate_nonempty("repository ID", &self.repo_id)?;
        validate_nonempty("revision", &self.revision)?;
        validate_repo_id(&self.repo_id)?;
        validate_revision(&self.revision)?;

        if let Some(token) = &self.token {
            validate_nonempty("token", token)?;
        }
        if let ModelKind::Gguf { filename } = &self.kind {
            validate_root_filename(filename, "GGUF filename")?;
            if !filename.ends_with(".gguf") {
                return Err(invalid("GGUF filename must end with lowercase `.gguf`"));
            }
            if is_split_gguf(filename) {
                return Err(invalid("split GGUF sets are not supported"));
            }
        }
        Ok(())
    }

    fn cache(&self) -> Cache {
        self.cache_dir
            .clone()
            .map(Cache::new)
            .unwrap_or_else(Cache::from_env)
    }

    fn api_builder(&self, cache: Cache) -> ApiBuilder {
        let builder = ApiBuilder::from_cache(cache).with_progress(self.progress);
        match &self.token {
            Some(token) => builder.with_token(Some(token.clone())),
            None => builder,
        }
    }

    fn requested_repo(&self) -> Repo {
        Repo::with_revision(self.repo_id.clone(), RepoType::Model, self.revision.clone())
    }

    fn resolve_gguf_online(
        &self,
        cache: Cache,
        filename: &str,
    ) -> Result<PathBuf, HuggingFaceError> {
        let api = self
            .api_builder(cache)
            .build()
            .map_err(|error| hub(format!("could not create API: {error}")))?;
        let path = api
            .repo(self.requested_repo())
            .get(filename)
            .map_err(|error| {
                hub(format!(
                    "could not resolve `{filename}` from `{}` at `{}`: {error}",
                    self.repo_id, self.revision
                ))
            })?;
        verify_gguf_path(&path, filename)?;
        Ok(path)
    }

    fn resolve_safetensors_online(&self, cache: Cache) -> Result<PathBuf, HuggingFaceError> {
        let api = self
            .api_builder(cache.clone())
            .build()
            .map_err(|error| hub(format!("could not create API: {error}")))?;
        let info = api.repo(self.requested_repo()).info().map_err(|error| {
            hub(format!(
                "could not read metadata for `{}` at `{}`: {error}",
                self.repo_id, self.revision
            ))
        })?;
        let sha = info.sha.trim();
        if sha.is_empty() {
            return Err(incomplete("repository metadata has an empty commit SHA"));
        }
        validate_root_filename(sha, "repository metadata SHA")
            .map_err(|_| incomplete("repository metadata has an unsafe SHA"))?;

        let available = sibling_names(&info);
        let bootstrap = plan_safetensors(&available)?;
        let pinned_repo =
            Repo::with_revision(self.repo_id.clone(), RepoType::Model, sha.to_owned());
        let pinned = api.repo(pinned_repo.clone());
        let index = if bootstrap.indexed {
            let path = pinned.get(SAFETENSORS_INDEX).map_err(|error| {
                hub(format!(
                    "could not resolve `{SAFETENSORS_INDEX}` for `{}` at `{sha}`: {error}",
                    self.repo_id
                ))
            })?;
            let bytes = fs::read(&path).map_err(|error| {
                io_error(format!("could not read `{}`: {error}", path.display()))
            })?;
            Some((path, bytes))
        } else {
            None
        };
        let (plan, index_path) = match index {
            Some((path, bytes)) => (plan_indexed_safetensors(&available, &bytes)?, Some(path)),
            None => (bootstrap, None),
        };

        let mut paths = HashMap::new();
        if let Some(path) = index_path {
            paths.insert(SAFETENSORS_INDEX.to_owned(), path);
        }
        for filename in &plan.files {
            if paths.contains_key(filename) {
                continue;
            }
            let path = pinned.get(filename).map_err(|error| {
                hub(format!(
                    "could not resolve `{filename}` for `{}` at `{sha}`: {error}",
                    self.repo_id
                ))
            })?;
            paths.insert(filename.clone(), path);
        }

        let snapshot = verify_snapshot_paths(&paths, sha)?;
        cache
            .repo(self.requested_repo())
            .create_ref(sha)
            .map_err(|error| {
                io_error(format!(
                    "could not update cache ref `{}` for `{}`: {error}",
                    self.revision, self.repo_id
                ))
            })?;
        Ok(snapshot)
    }

    fn resolve_offline(&self, cache: &Cache) -> Result<PathBuf, HuggingFaceError> {
        let repo = cache.repo(self.requested_repo());
        match &self.kind {
            ModelKind::Gguf { filename } => {
                let path = repo.get(filename).ok_or_else(|| {
                    cache_miss(format!(
                        "`{filename}` for `{}` at `{}` is not cached",
                        self.repo_id, self.revision
                    ))
                })?;
                let snapshot = snapshot_for_cached_revision(cache, &self.requested_repo())?;
                verify_gguf_path(&path, filename)?;
                if path.parent() != Some(snapshot.as_path()) {
                    return Err(incomplete(format!(
                        "resolved `{filename}` does not belong to the requested revision snapshot"
                    )));
                }
                Ok(path)
            }
            ModelKind::Safetensors => {
                let snapshot = snapshot_for_cached_revision(cache, &self.requested_repo())?;
                let available = cached_root_files(&snapshot)?;
                let bootstrap = plan_safetensors(&available)?;
                let plan = if bootstrap.indexed {
                    let index_path = snapshot.join(SAFETENSORS_INDEX);
                    let bytes = fs::read(&index_path).map_err(|error| {
                        io_error(format!(
                            "could not read `{}`: {error}",
                            index_path.display()
                        ))
                    })?;
                    plan_indexed_safetensors(&available, &bytes)?
                } else {
                    bootstrap
                };

                let paths = plan
                    .files
                    .iter()
                    .map(|filename| (filename.clone(), snapshot.join(filename)))
                    .collect::<HashMap<_, _>>();
                verify_snapshot_paths(&paths, snapshot_sha(&snapshot)?)
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct SafetensorsPlan {
    files: Vec<String>,
    indexed: bool,
}

fn plan_safetensors(available: &BTreeSet<String>) -> Result<SafetensorsPlan, HuggingFaceError> {
    for required in [CONFIG, TOKENIZER] {
        if !available.contains(required) {
            return Err(incomplete(format!("required `{required}` is missing")));
        }
    }

    let mut files = vec![CONFIG.to_owned(), TOKENIZER.to_owned()];
    if available.contains(TOKENIZER_CONFIG) {
        files.push(TOKENIZER_CONFIG.to_owned());
    }
    if available.contains(SAFETENSORS) {
        files.push(SAFETENSORS.to_owned());
        Ok(SafetensorsPlan {
            files,
            indexed: false,
        })
    } else if available.contains(SAFETENSORS_INDEX) {
        files.push(SAFETENSORS_INDEX.to_owned());
        Ok(SafetensorsPlan {
            files,
            indexed: true,
        })
    } else {
        Err(incomplete(format!(
            "neither `{SAFETENSORS}` nor `{SAFETENSORS_INDEX}` is present"
        )))
    }
}

fn plan_indexed_safetensors(
    available: &BTreeSet<String>,
    bytes: &[u8],
) -> Result<SafetensorsPlan, HuggingFaceError> {
    let mut plan = plan_safetensors(available)?;
    if !plan.indexed {
        return Ok(plan);
    }

    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| incomplete(format!("`{SAFETENSORS_INDEX}` is malformed JSON: {error}")))?;
    let weight_map = value
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| incomplete(format!("`{SAFETENSORS_INDEX}` has no object `weight_map`")))?;
    if weight_map.is_empty() {
        return Err(incomplete(format!(
            "`{SAFETENSORS_INDEX}` has an empty `weight_map`"
        )));
    }

    let mut shards = BTreeSet::new();
    for value in weight_map.values() {
        let shard = value.as_str().ok_or_else(|| {
            incomplete(format!(
                "`{SAFETENSORS_INDEX}` contains a non-string shard path"
            ))
        })?;
        validate_root_filename(shard, "Safetensors shard")
            .map_err(|error| incomplete(error.to_string()))?;
        if !shard.ends_with(".safetensors") {
            return Err(incomplete(format!(
                "indexed shard `{shard}` must end with `.safetensors`"
            )));
        }
        if !available.contains(shard) {
            return Err(incomplete(format!(
                "indexed shard `{shard}` is missing from repository metadata or cache"
            )));
        }
        shards.insert(shard.to_owned());
    }
    plan.files.extend(shards);
    Ok(plan)
}

fn sibling_names(info: &RepoInfo) -> BTreeSet<String> {
    info.siblings
        .iter()
        .filter(|sibling| {
            validate_root_filename(&sibling.rfilename, "repository metadata filename").is_ok()
        })
        .map(|sibling| sibling.rfilename.clone())
        .collect()
}

fn cached_root_files(snapshot: &Path) -> Result<BTreeSet<String>, HuggingFaceError> {
    let entries = fs::read_dir(snapshot).map_err(|error| {
        io_error(format!(
            "could not read cached snapshot `{}`: {error}",
            snapshot.display()
        ))
    })?;
    let mut files = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            io_error(format!(
                "could not inspect cached snapshot `{}`: {error}",
                snapshot.display()
            ))
        })?;
        if entry.path().is_file() {
            if let Some(filename) = entry.file_name().to_str() {
                files.insert(filename.to_owned());
            }
        }
    }
    Ok(files)
}

fn snapshot_for_cached_revision(cache: &Cache, repo: &Repo) -> Result<PathBuf, HuggingFaceError> {
    let cache_repo = cache.repo(repo.clone());
    let ref_path = cache
        .path()
        .join(repo.folder_name())
        .join("refs")
        .join(repo.revision());
    let sha = fs::read_to_string(&ref_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            cache_miss(format!(
                "revision `{}` for `{}` has no cache ref",
                repo.revision(),
                repo.folder_name()
            ))
        } else {
            io_error(format!(
                "could not read cache ref `{}`: {error}",
                ref_path.display()
            ))
        }
    })?;
    let sha = sha.trim();
    if sha.is_empty() {
        return Err(incomplete(format!(
            "cache ref `{}` has an empty SHA",
            ref_path.display()
        )));
    }
    validate_root_filename(sha, "cached revision SHA")
        .map_err(|_| incomplete("cached revision has an unsafe SHA"))?;
    let snapshot = cache_repo.pointer_path(sha);
    if !snapshot.is_dir() {
        return Err(cache_miss(format!(
            "snapshot `{}` is not cached",
            snapshot.display()
        )));
    }
    Ok(snapshot)
}

fn verify_gguf_path(path: &Path, filename: &str) -> Result<(), HuggingFaceError> {
    if !path.is_file() {
        return Err(incomplete(format!(
            "resolved `{filename}` is not a file at `{}`",
            path.display()
        )));
    }
    if path.file_name().and_then(|name| name.to_str()) != Some(filename) {
        return Err(incomplete(format!(
            "resolved GGUF path does not match requested filename `{filename}`"
        )));
    }
    let snapshot = path
        .parent()
        .ok_or_else(|| incomplete(format!("resolved `{filename}` has no snapshot directory")))?;
    if snapshot
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        != Some("snapshots")
    {
        return Err(incomplete(format!(
            "resolved `{filename}` is not directly under a `snapshots` directory"
        )));
    }
    Ok(())
}

fn snapshot_sha(snapshot: &Path) -> Result<&str, HuggingFaceError> {
    snapshot
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| incomplete("cached snapshot has a non-UTF-8 SHA"))
}

fn verify_snapshot_paths(
    paths: &HashMap<String, PathBuf>,
    sha: &str,
) -> Result<PathBuf, HuggingFaceError> {
    if paths.is_empty() {
        return Err(incomplete("no snapshot files were resolved"));
    }
    let mut common = None;
    for (filename, path) in paths {
        if !path.is_file() {
            return Err(incomplete(format!(
                "resolved `{filename}` is not a file at `{}`",
                path.display()
            )));
        }
        let parent = path.parent().ok_or_else(|| {
            incomplete(format!("resolved `{filename}` has no snapshot directory"))
        })?;
        let valid_layout = parent.file_name().and_then(|name| name.to_str()) == Some(sha)
            && parent
                .parent()
                .and_then(Path::file_name)
                .and_then(|name| name.to_str())
                == Some("snapshots");
        if !valid_layout {
            return Err(incomplete(format!(
                "resolved `{filename}` is outside `snapshots/{sha}`"
            )));
        }
        match &common {
            Some(expected) if expected != parent => {
                return Err(incomplete("resolved files belong to different snapshots"));
            }
            None => common = Some(parent.to_owned()),
            _ => {}
        }
    }
    common.ok_or_else(|| incomplete("no snapshot directory was resolved"))
}

fn validate_nonempty(field: &str, value: &str) -> Result<(), HuggingFaceError> {
    if value.trim().is_empty() {
        Err(invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_repo_id(repo_id: &str) -> Result<(), HuggingFaceError> {
    validate_repo_relative_path(repo_id, "repository ID")?;
    if repo_id.split('/').count() > 2 {
        return Err(invalid("repository ID must be `name` or `namespace/name`"));
    }
    Ok(())
}

fn validate_revision(revision: &str) -> Result<(), HuggingFaceError> {
    validate_repo_relative_path(revision, "revision")
}

fn validate_root_filename(filename: &str, field: &str) -> Result<(), HuggingFaceError> {
    validate_repo_relative_path(filename, field)?;
    if Path::new(filename).components().count() != 1 {
        return Err(invalid(format!("{field} must be a root-level filename")));
    }
    Ok(())
}

fn validate_repo_relative_path(path: &str, field: &str) -> Result<(), HuggingFaceError> {
    validate_nonempty(field, path)?;
    if path.contains('\\')
        || path.contains('\0')
        || path.contains(':')
        || path.starts_with('~')
        || path.split('/').any(str::is_empty)
    {
        return Err(invalid(format!(
            "{field} must use portable repository-relative path syntax"
        )));
    }
    let path = Path::new(path);
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(invalid(format!(
            "{field} must not be absolute or contain `.` or `..` components"
        )));
    }
    for component in path.components() {
        let Component::Normal(component) = component else {
            unreachable!("non-normal components were rejected above");
        };
        let component = component.to_string_lossy();
        if component.ends_with([' ', '.'])
            || component.chars().any(|value| {
                value.is_control() || matches!(value, '<' | '>' | '"' | '|' | '?' | '*')
            })
        {
            return Err(invalid(format!(
                "{field} contains characters that are not portable path syntax"
            )));
        }
    }
    Ok(())
}

fn is_split_gguf(filename: &str) -> bool {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    let Some((prefix, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, part)) = prefix.rsplit_once('-') else {
        return false;
    };
    !part.is_empty()
        && !total.is_empty()
        && part.bytes().all(|value| value.is_ascii_digit())
        && total.bytes().all(|value| value.is_ascii_digit())
}

fn invalid(message: impl Into<String>) -> HuggingFaceError {
    HuggingFaceError::InvalidInput {
        message: message.into(),
    }
}

fn cache_miss(message: impl Into<String>) -> HuggingFaceError {
    HuggingFaceError::CacheMiss {
        message: message.into(),
    }
}

fn incomplete(message: impl Into<String>) -> HuggingFaceError {
    HuggingFaceError::Incomplete {
        message: message.into(),
    }
}

fn hub(message: impl Into<String>) -> HuggingFaceError {
    HuggingFaceError::Hub {
        message: message.into(),
    }
}

fn io_error(message: impl Into<String>) -> HuggingFaceError {
    HuggingFaceError::Io {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::Siblings;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    const REPO: &str = "owner/model";
    const REVISION: &str = "release";
    const SHA: &str = "0123456789abcdef";

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("vllm-cpp-hf-{}-{id}", std::process::id()));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn names(values: &[&str]) -> BTreeSet<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn assert_incomplete_contains<T: fmt::Debug>(
        result: Result<T, HuggingFaceError>,
        expected: &str,
    ) {
        match result {
            Err(HuggingFaceError::Incomplete { message }) => assert!(
                message.contains(expected),
                "expected `{message}` to contain `{expected}`"
            ),
            other => panic!("expected incomplete error containing `{expected}`, got {other:?}"),
        }
    }

    fn cache_fixture_at(revision: &str, files: &[(&str, &[u8])]) -> TempDir {
        let temp = TempDir::new();
        let cache = Cache::new(temp.0.clone());
        let repo = Repo::with_revision(REPO.to_owned(), RepoType::Model, revision.to_owned());
        let cache_repo = cache.repo(repo);
        cache_repo.create_ref(SHA).unwrap();
        let snapshot = cache_repo.pointer_path(SHA);
        fs::create_dir_all(&snapshot).unwrap();
        for (filename, contents) in files {
            fs::write(snapshot.join(filename), contents).unwrap();
        }
        temp
    }

    fn cache_fixture(files: &[(&str, &[u8])]) -> TempDir {
        cache_fixture_at(DEFAULT_REVISION, files)
    }

    #[test]
    fn validates_inputs_and_rejects_split_gguf() {
        let cases = [
            HuggingFaceModel::gguf("", "model.gguf"),
            HuggingFaceModel::gguf(REPO, "model.gguf").revision(""),
            HuggingFaceModel::gguf(REPO, "model.gguf").revision("../main"),
            HuggingFaceModel::gguf("owner/model/extra", "model.gguf"),
            HuggingFaceModel::gguf(REPO, "/model.gguf"),
            HuggingFaceModel::gguf(REPO, "nested/model.gguf"),
            HuggingFaceModel::gguf(REPO, "model.GGUF"),
            HuggingFaceModel::gguf(REPO, "model-00001-of-00002.gguf"),
            HuggingFaceModel::gguf(REPO, "model-1-of-2.gguf"),
            HuggingFaceModel::gguf(REPO, "model?.gguf"),
        ];
        for model in cases {
            assert!(matches!(
                model.validate(),
                Err(HuggingFaceError::InvalidInput { .. })
            ));
        }
        assert!(HuggingFaceModel::gguf(REPO, "model.gguf")
            .revision("refs/pr/1")
            .validate()
            .is_ok());
    }

    #[test]
    fn defaults_to_main_and_accepts_revision_override() {
        let default_gguf = HuggingFaceModel::gguf(REPO, "model.gguf");
        let default_safetensors = HuggingFaceModel::safetensors(REPO);
        assert_eq!(default_gguf.revision, DEFAULT_REVISION);
        assert_eq!(default_safetensors.revision, DEFAULT_REVISION);

        let pinned = default_safetensors.revision(REVISION);
        assert_eq!(pinned.revision, REVISION);
        assert_eq!(pinned.requested_repo().revision(), REVISION);
    }

    #[test]
    fn debug_redacts_explicit_token() {
        let model = HuggingFaceModel::safetensors(REPO)
            .revision(REVISION)
            .token("hf_secret_value");
        let debug = format!("{model:?}");
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("hf_secret_value"));
    }

    #[test]
    fn plans_unsharded_and_indexed_snapshots() {
        let unsharded = names(&[CONFIG, TOKENIZER, TOKENIZER_CONFIG, SAFETENSORS]);
        assert_eq!(
            plan_safetensors(&unsharded).unwrap(),
            SafetensorsPlan {
                files: vec![CONFIG, TOKENIZER, TOKENIZER_CONFIG, SAFETENSORS]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
                indexed: false,
            }
        );

        let indexed = names(&[
            CONFIG,
            TOKENIZER,
            SAFETENSORS_INDEX,
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]);
        let bytes = br#"{"weight_map":{"a":"model-00002-of-00002.safetensors","b":"model-00001-of-00002.safetensors","c":"model-00002-of-00002.safetensors"}}"#;
        let plan = plan_indexed_safetensors(&indexed, bytes).unwrap();
        assert_eq!(
            plan.files,
            vec![
                CONFIG,
                TOKENIZER,
                SAFETENSORS_INDEX,
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ]
        );
    }

    #[test]
    fn rejects_incomplete_or_malformed_safetensors_metadata() {
        let missing_core = names(&[CONFIG, SAFETENSORS]);
        assert!(matches!(
            plan_safetensors(&missing_core),
            Err(HuggingFaceError::Incomplete { .. })
        ));

        let indexed = names(&[CONFIG, TOKENIZER, SAFETENSORS_INDEX, "part.safetensors"]);
        for (bytes, expected) in [
            (br#"not json"#.as_slice(), "malformed JSON"),
            (br#"{}"#.as_slice(), "has no object `weight_map`"),
            (br#"{"weight_map":{}}"#.as_slice(), "empty `weight_map`"),
            (
                br#"{"weight_map":{"a":3}}"#.as_slice(),
                "non-string shard path",
            ),
            (
                br#"{"weight_map":{"a":"../part.safetensors"}}"#.as_slice(),
                "must not be absolute or contain `.` or `..` components",
            ),
            (
                br#"{"weight_map":{"a":"/part.safetensors"}}"#.as_slice(),
                "must use portable repository-relative path syntax",
            ),
            (
                br#"{"weight_map":{"a":"nested/part.safetensors"}}"#.as_slice(),
                "must be a root-level filename",
            ),
            (
                br#"{"weight_map":{"a":"missing.safetensors"}}"#.as_slice(),
                "indexed shard `missing.safetensors` is missing",
            ),
            (
                br#"{"weight_map":{"a":"part.bin"}}"#.as_slice(),
                "must end with `.safetensors`",
            ),
        ] {
            assert_incomplete_contains(plan_indexed_safetensors(&indexed, bytes), expected);
        }
    }

    #[test]
    fn ignores_unrelated_unsafe_safetensors_siblings() {
        let info = RepoInfo {
            sha: SHA.to_owned(),
            siblings: [
                CONFIG,
                TOKENIZER,
                SAFETENSORS,
                "../junk.json",
                "/junk.json",
                "nested/junk.json",
            ]
            .into_iter()
            .map(|rfilename| Siblings {
                rfilename: rfilename.to_owned(),
            })
            .collect(),
        };

        let available = sibling_names(&info);
        assert_eq!(available, names(&[CONFIG, TOKENIZER, SAFETENSORS]));
        assert!(plan_safetensors(&available).is_ok());

        let unsafe_required = RepoInfo {
            sha: SHA.to_owned(),
            siblings: ["../config.json", TOKENIZER, SAFETENSORS]
                .into_iter()
                .map(|rfilename| Siblings {
                    rfilename: rfilename.to_owned(),
                })
                .collect(),
        };
        assert_incomplete_contains(
            plan_safetensors(&sibling_names(&unsafe_required)),
            "required `config.json` is missing",
        );
    }

    #[test]
    fn resolves_complete_offline_unsharded_cache() {
        let fixture = cache_fixture(&[
            (CONFIG, b"{}"),
            (TOKENIZER, b"{}"),
            (TOKENIZER_CONFIG, b"{}"),
            (SAFETENSORS, b"weights"),
        ]);
        let resolved = HuggingFaceModel::safetensors(REPO)
            .cache_dir(&fixture.0)
            .offline(true)
            .resolve()
            .unwrap();
        assert_eq!(resolved.file_name().unwrap(), SHA);
        assert_eq!(resolved.parent().unwrap().file_name().unwrap(), "snapshots");
    }

    #[test]
    fn resolves_complete_offline_indexed_cache() {
        let index = br#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#;
        let fixture = cache_fixture_at(
            REVISION,
            &[
                (CONFIG, b"{}"),
                (TOKENIZER, b"{}"),
                (SAFETENSORS_INDEX, index),
                ("model-00001-of-00002.safetensors", b"one"),
                ("model-00002-of-00002.safetensors", b"two"),
            ],
        );
        assert!(HuggingFaceModel::safetensors(REPO)
            .revision(REVISION)
            .cache_dir(&fixture.0)
            .offline(true)
            .resolve()
            .is_ok());
    }

    #[test]
    fn distinguishes_offline_cache_miss_and_incomplete_snapshot() {
        let empty = TempDir::new();
        let miss = HuggingFaceModel::safetensors(REPO)
            .cache_dir(&empty.0)
            .offline(true)
            .resolve();
        assert!(matches!(miss, Err(HuggingFaceError::CacheMiss { .. })));

        let partial = cache_fixture(&[(CONFIG, b"{}"), (TOKENIZER, b"{}")]);
        let incomplete = HuggingFaceModel::safetensors(REPO)
            .cache_dir(&partial.0)
            .offline(true)
            .resolve();
        assert!(matches!(
            incomplete,
            Err(HuggingFaceError::Incomplete { .. })
        ));

        let index = br#"{"weight_map":{"a":"missing.safetensors"}}"#;
        let missing_shard = cache_fixture(&[
            (CONFIG, b"{}"),
            (TOKENIZER, b"{}"),
            (SAFETENSORS_INDEX, index),
        ]);
        assert_incomplete_contains(
            HuggingFaceModel::safetensors(REPO)
                .cache_dir(&missing_shard.0)
                .offline(true)
                .resolve(),
            "indexed shard `missing.safetensors` is missing",
        );
    }

    #[test]
    fn resolves_offline_gguf_and_reports_missing_file() {
        let fixture = cache_fixture(&[("model.gguf", b"gguf")]);
        let resolved = HuggingFaceModel::gguf(REPO, "model.gguf")
            .cache_dir(&fixture.0)
            .offline(true)
            .resolve()
            .unwrap();
        assert_eq!(resolved.file_name().unwrap(), "model.gguf");

        let missing = HuggingFaceModel::gguf(REPO, "missing.gguf")
            .cache_dir(&fixture.0)
            .offline(true)
            .resolve();
        assert!(matches!(missing, Err(HuggingFaceError::CacheMiss { .. })));
    }

    #[test]
    fn verifies_gguf_filename_and_snapshot_layout() {
        let fixture = cache_fixture(&[("model.gguf", b"gguf")]);
        let path = Cache::new(fixture.0.clone())
            .repo(Repo::with_revision(
                REPO.to_owned(),
                RepoType::Model,
                DEFAULT_REVISION.to_owned(),
            ))
            .get("model.gguf")
            .unwrap();
        assert!(verify_gguf_path(&path, "model.gguf").is_ok());
        assert_incomplete_contains(
            verify_gguf_path(&path, "other.gguf"),
            "does not match requested filename",
        );

        let outside = fixture.0.join("outside.gguf");
        fs::write(&outside, b"gguf").unwrap();
        assert_incomplete_contains(
            verify_gguf_path(&outside, "outside.gguf"),
            "not directly under a `snapshots` directory",
        );
    }

    #[test]
    fn rejects_mixed_snapshot_paths() {
        let temp = TempDir::new();
        let first = temp.0.join("snapshots").join(SHA);
        let second = temp.0.join("snapshots").join("other");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        fs::write(first.join(CONFIG), b"{}").unwrap();
        fs::write(second.join(TOKENIZER), b"{}").unwrap();
        let paths = HashMap::from([
            (CONFIG.to_owned(), first.join(CONFIG)),
            (TOKENIZER.to_owned(), second.join(TOKENIZER)),
        ]);
        assert!(matches!(
            verify_snapshot_paths(&paths, SHA),
            Err(HuggingFaceError::Incomplete { .. })
        ));
    }
}
