# In Identity Mappings in Deep Residual Networks
# dataset   network      baseline   pre-activation
# CIFAR-10  ResNet-110   6.61       6.37
#           ResNet-164   5.93       5.46
#           ResNet-1001  7.61       4.92
# CIFAR-100 ResNet-164   25.16      24.33
#           ResNet-1001  27.82      22.71
set -e

EXAMPLE=examples/cifar100
DATA=data/cifar100
DBTYPE=lmdb
LOG=examples/cifar100/log
SNAPSHOT=examples/cifar100/snapshot

echo "Creating $DBTYPE..."

rm -rf $EXAMPLE/cifar100_train_$DBTYPE $EXAMPLE/cifar100_test_$DBTYPE

./build/examples/cifar100/convert_cifar100_data.bin $DATA $EXAMPLE $DBTYPE

echo "Computing image mean..."

./build/tools/compute_image_mean -backend=$DBTYPE \
  $EXAMPLE/cifar100_train_$DBTYPE $EXAMPLE/mean.binaryproto

if [ -d "$LOG" ]; then
    rm -rf $LOG
    echo "Remove Dir : $LOG"
fi
mkdir $LOG

if [ -d "$SNAPSHOT" ]; then
    rm -rf $SNAPSHOT
    echo "Remove Dir : $SNAPSHOT"
fi
mkdir $SNAPSHOT
python ./examples/cifar100/PadCifar100.py

echo "Done."
