TRAIN_TEST_SPLIT=navtest
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH

TRAIN_TEST_SPLIT=navhard_two_stage
CACHE_PATH=$NAVSIM_EXP_ROOT/navhard_two_stage_metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
metric_cache_path=$CACHE_PATH
