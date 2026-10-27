include!("../examples/common/mod.rs");

#[cfg(test)]
mod tests {
    use super::*;

    fn os(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_local_aliases() {
        assert_eq!(
            parse_model_source("example", os(&["model.gguf"])).unwrap(),
            ModelSource::Local("model.gguf".into())
        );
        assert_eq!(
            parse_model_source("example", os(&["local", "model"])).unwrap(),
            ModelSource::Local("model".into())
        );
    }

    #[test]
    fn parses_hugging_face_sources_with_optional_revisions() {
        assert_eq!(
            parse_model_source("example", os(&["hf-gguf", "owner/model", "model.gguf"])).unwrap(),
            ModelSource::Gguf {
                repo: "owner/model".to_owned(),
                filename: "model.gguf".to_owned(),
                revision: None,
            }
        );
        assert_eq!(
            parse_model_source("example", os(&["hf-safetensors", "owner/model"])).unwrap(),
            ModelSource::Safetensors {
                repo: "owner/model".to_owned(),
                revision: None,
            }
        );
        assert_eq!(
            parse_model_source(
                "example",
                os(&[
                    "hf-gguf",
                    "owner/model",
                    "model.gguf",
                    "--revision",
                    "release",
                ]),
            )
            .unwrap(),
            ModelSource::Gguf {
                repo: "owner/model".to_owned(),
                filename: "model.gguf".to_owned(),
                revision: Some("release".to_owned()),
            }
        );
        assert_eq!(
            parse_model_source(
                "example",
                os(&["hf-safetensors", "owner/model", "--revision", "release",]),
            )
            .unwrap(),
            ModelSource::Safetensors {
                repo: "owner/model".to_owned(),
                revision: Some("release".to_owned()),
            }
        );
    }

    #[test]
    fn rejects_missing_extra_duplicate_and_misordered_arguments() {
        for args in [
            os(&[]),
            os(&[""]),
            os(&["local"]),
            os(&["local", ""]),
            os(&["model", "extra"]),
            os(&["hf-gguf", "owner/model"]),
            os(&["hf-gguf", "owner/model", ""]),
            os(&["hf-safetensors", ""]),
            os(&["hf-safetensors", "owner/model", "release"]),
            os(&["hf-safetensors", "owner/model", "--revision"]),
            os(&["hf-safetensors", "owner/model", "--revision", ""]),
            os(&[
                "hf-safetensors",
                "owner/model",
                "--revision",
                "one",
                "--revision",
                "two",
            ]),
            os(&[
                "hf-gguf",
                "--revision",
                "release",
                "owner/model",
                "model.gguf",
            ]),
        ] {
            assert!(parse_model_source("example", args).is_err());
        }
    }
}
