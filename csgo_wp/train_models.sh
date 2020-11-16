python train.py --transform channels --n-epochs 10 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --learning-rate 0.00001 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,1,3,1,0,2,1,0" --learning-rate 0.00001 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type res --learning-rate 0.00001 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --hidden-sizes 200,1000,500,200,50 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --learning-rate 0.00001 --hidden-sizes 200,1000,500,200,50 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-size 2 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-size 4 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-size 8 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-size 16 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-size 32 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-size 64 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-norm True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --dropout True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --batch-norm True --dropout True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,6,5,1,0,1,1,0" >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,1,3,1,0,2,1,0" --batch-norm True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,1,3,1,0,2,1,0" --dropout True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,1,3,1,0,2,1,0" --batch-norm True --dropout True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type res --batch-norm True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type res --dropout True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type res --batch-norm True --dropout True >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --activation LeakyReLU >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,1,3,1,0,2,1,0" --activation LeakyReLU >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type res --activation LeakyReLU >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,3,3,1,0,2,1,0|3,1,3,1,0,2,1,0|1,1,3,1,0,2,1,0" >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,3,3,1,0,2,1,0|3,1,3,1,0,2,1,0" >> training_log.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" >> training_log.log 2>&1
python train.py --transform channels --n-epochs 100 >> training_log.log 2>&1
python train.py --transform channels --n-epochs 100 --learning-rate 0.00001 >> training_log.log 2>&1

# FC
python train.py --transform channels --n-epochs 100 --activation LeakyReLU --batch-norm True --dropout True --learning-rate 0.00001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --activation LeakyReLU --batch-norm True --dropout True --learning-rate 0.000001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --activation LeakyReLU --batch-norm True --dropout True --learning-rate 0.000005 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --activation LeakyReLU --batch-norm True --dropout True --learning-rate 0.0000005 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --activation LeakyReLU --batch-norm True --dropout True --batch-size 64 --learning-rate 0.00001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --activation LeakyReLU --batch-norm True --dropout True --batch-size 96 --learning-rate 0.00001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --activation LeakyReLU --batch-norm True --dropout True --batch-size 128 --learning-rate 0.00001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --activation LeakyReLU --batch-norm True --dropout True --batch-size 192 --learning-rate 0.00001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --activation LeakyReLU --batch-norm True --dropout True --batch-size 256 --learning-rate 0.00001 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --activation LeakyReLU --batch-norm True --dropout True --batch-size 512 --learning-rate 0.00001 >> training_log_8.log 2>&1

# CNN
python train.py --transform channels --n-epochs 100 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00005 --activation LeakyReLU --batch-norm True >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.000001 --activation LeakyReLU --batch-norm True >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.000005 --activation LeakyReLU --batch-norm True >> training_log_8.log 2>&1

python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True --batch-size 64 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True --batch-size 96 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True --batch-size 128 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True --batch-size 192 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True --batch-size 256 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True --batch-size 512 >> training_log_8.log 2>&1

python train.py --transform channels --n-epochs 25 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,3,3,1,0,2,1,0|3,1,2,1,0,2,1,0|1,1,1,1,0,1,1,0" >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 10 --model-type cnn --cnn-options "6,3,3,1,0,2,1,0|3,1,2,1,0,2,1,0" >> training_log_8.log 2>&1

# ResNet
python train.py --transform channels --n-epochs 100 --model-type res --learning-rate 0.00001 --dropout True --activation LeakyReLU >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type res --learning-rate 0.00005 --dropout True --activation LeakyReLU >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type res --learning-rate 0.000001 --dropout True --activation LeakyReLU >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type res --learning-rate 0.000005 --dropout True --activation LeakyReLU >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type res --activation LeakyReLU --learning-rate 0.00001 --dropout True --batch-size 64 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type res --activation LeakyReLU --learning-rate 0.00001 --dropout True --batch-size 96 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type res --activation LeakyReLU --learning-rate 0.00001 --dropout True --batch-size 128 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type res --activation LeakyReLU --learning-rate 0.00001 --dropout True --batch-size 192 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type res --activation LeakyReLU --learning-rate 0.00001 --dropout True --batch-size 256 >> training_log_8.log 2>&1
python train.py --transform channels --n-epochs 25 --model-type res --activation LeakyReLU --learning-rate 0.00001 --dropout True --batch-size 512 >> training_log_8.log 2>&1

# no early stopping
python train.py --transform channels --n-epochs 100 --activation LeakyReLU --batch-norm True --dropout True --learning-rate 0.00001 >> training_log_15.log 2>&1
python train.py --transform channels --n-epochs 100 --activation LeakyReLU --batch-norm True --dropout True --batch-size 64 --learning-rate 0.00001 >> training_log_15.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True >> training_log_15.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type cnn --cnn-options "6,6,1,1,0,1,1,0|6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --learning-rate 0.00001 --activation LeakyReLU --batch-norm True >> training_log_15.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type res --learning-rate 0.00001 --dropout True --activation LeakyReLU >> training_log_15.log 2>&1

# lrcnn
python train.py --transform channels --n-epochs 100 --model-type lrcnn --cnn-options "4,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --activation LeakyReLU --learning-rate 0.00001 --batch-size 64 >> training_log_lrcnn.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type lrcnn --cnn-options "4,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --activation LeakyReLU --learning-rate 0.00001 --batch-size 96 >> training_log_lrcnn.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type lrcnn --cnn-options "4,15,1,1,0,1,1,0|15,6,5,1,0,1,1,0" --activation LeakyReLU --learning-rate 0.00001 --batch-size 64 >> training_log_lrcnn.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type lrcnn --cnn-options "4,6,1,1,0,1,1,0|6,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --activation LeakyReLU --learning-rate 0.00001 --batch-size 64 >> training_log_lrcnn.log 2>&1
python train.py --transform channels --n-epochs 100 --model-type lrcnn --cnn-options "4,6,1,1,0,1,1,0|6,6,5,1,0,1,1,0" --activation LeakyReLU --learning-rate 0.000001 --batch-size 96 >> training_log_lrcnn_2.log 2>&1