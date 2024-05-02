nsys profile \
--output profile/qsvm_cuTN_ZZ-liner-r1-2q_100s_rtx4090_blocking-F \
--cuda-memory-usage true \
--cuda-um-cpu-page-faults false \
--cuda-um-gpu-page-faults false \
--force-overwrite true \
--trace-fork-before-exec true \
--gpu-metrics-device none \
python qsvm_cuTN_profile.py