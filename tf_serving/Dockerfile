FROM tensorflow/serving:2.13.1-gpu
EXPOSE 8500 8501
ENV TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2"
CMD ["tensorflow_model_server", "--model_config_file=/etc/config/models.conf","--rest_api_timeout_in_ms=100000","--model_config_file_poll_wait_seconds=60","--file_system_poll_wait_seconds=60", "--xla_cpu_compilation_enabled=true"]
