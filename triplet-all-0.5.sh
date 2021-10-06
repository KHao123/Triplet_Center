# python main.py --dataset_name sd198 --iterNo 0 --cuda_device 2 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 20 --n_samples_per_class 5 --TripletSelector RandomNegativeTripletSelector --margin 0.5 --batch_size 16
# python main.py --dataset_name sd198 --iterNo 1 --cuda_device 3 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 20 --n_samples_per_class 5 --TripletSelector RandomNegativeTripletSelector --margin 0.5 --batch_size 16
# python main.py --dataset_name sd198 --iterNo 2 --cuda_device 3 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 20 --n_samples_per_class 5 --TripletSelector RandomNegativeTripletSelector --margin 0.5 --batch_size 16
# python main.py --dataset_name sd198 --iterNo 3 --cuda_device 3 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 20 --n_samples_per_class 5 --TripletSelector RandomNegativeTripletSelector --margin 0.5 --batch_size 16
# python main.py --dataset_name sd198 --iterNo 4 --cuda_device 3 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 20 --n_samples_per_class 5 --TripletSelector RandomNegativeTripletSelector --margin 0.5 --batch_size 16

#baseline 没有normalize all sample
# rm -rf ~/triplet-center/run/result/Skin7/iterNo1/margin0.5_128d-embedding_AllTripletSelector
#图片归一化
python main.py --dataset_name skin --iterNo 1 --cuda_device 6 --EmbeddingMode t --dim 128 --n_epoch 200 --n_sample_classes 7 --n_samples_per_class 10 --TripletSelector AllTripletSelector --margin 0.5 --batch_size 16
# python main.py --dataset_name skin --iterNo 2 --cuda_device 4 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 3 --n_samples_per_class 5 --TripletSelector AllTripletSelector --margin 0.5 --batch_size 48
# python main.py --dataset_name skin --iterNo 3 --cuda_device 4 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 3 --n_samples_per_class 5 --TripletSelector AllTripletSelector --margin 0.5 --batch_size 48
# python main.py --dataset_name skin --iterNo 4 --cuda_device 4 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 3 --n_samples_per_class 5 --TripletSelector AllTripletSelector --margin 0.5 --batch_size 48
# python main.py --dataset_name skin --iterNo 5 --cuda_device 4 --EmbeddingMode t --dim 128 --n_epoch 250 --n_sample_classes 3 --n_samples_per_class 5 --TripletSelector AllTripletSelector --margin 0.5 --batch_size 48
