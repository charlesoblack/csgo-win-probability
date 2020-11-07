python train.py --n-epochs 10 >> training_log.log 2>&1
python train.py --n-epochs 10 --learning-rate 0.00001 >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --learning-rate 0.00001 >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type resnet --learning-rate 0.00001 >> training_log.log 2>&1
python train.py --n-epochs 10 --hidden-sizes 200,1000,500,200,50 >> training_log.log 2>&1
python train.py --n-epochs 10 --learning-rate 0.00001 --hidden-sizes 200,1000,500,200,50 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-size 2 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-size 4 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-size 8 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-size 16 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-size 32 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-size 64 >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-norm True >> training_log.log 2>&1
python train.py --n-epochs 10 --dropout True >> training_log.log 2>&1
python train.py --n-epochs 10 --batch-norm True --dropout True >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --batch-norm True >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --dropout True >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --batch-norm True --dropout True >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type res --batch-norm True >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type res --dropout True >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type res --batch-norm True --dropout True >> training_log.log 2>&1
python train.py --n-epochs 10 --activation LeakyReLU >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --activation LeakyReLU >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type res --activation LeakyReLU >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --cnn-options "1,3,3,1,0,2,1,0|3,1,3,1,0,2,1,0|1,1,3,1,0,2,1,0" >> training_log.log 2>&1
python train.py --n-epochs 10 --model-type cnn --cnn-options "1,3,3,1,0,2,1,0|3,1,3,1,0,2,1,0" >> training_log.log 2>&1
python train.py --n-epochs 100 >> training_log.log 2>&1
python train.py --n-epochs 100 --learning-rate 0.00001 >> training_log.log 2>&1
