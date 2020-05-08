import numpy as np
import pandas as pd
import os


# 'run/result/skinimage/iter1/triplet.../result.csv'
skin7_columns = ['MEL_p', 'MEL_r', 'MEL_f1', 'NV_p', 'NV_r', 'NV_f1', 'BCC_p', 'BCC_r',
       'BCC_f1', 'AKIEC_p', 'AKIEC_r', 'AKIEC_f1', 'BKL_p', 'BKL_r', 'BKL_f1',
       'DF_p', 'DF_r', 'DF_f1', 'VASC_p', 'VASC_r', 'VASC_f1', 'mean_p',
       'mean_r', 'mean_f1']

sd198_columns = ['small_mcp','small_mcr','small_mf1','mean_p','mean_r','mean_f1']
xray13_columns = ['Normal_p','Normal_r','Normal_f1','Lung Opacity_p','Lung Opacity_r','Lung Opacity_f1','‘No Lung Opacity/Not Normal_p','‘No Lung Opacity/Not Normal_r','‘No Lung Opacity/Not Normal_f1','mean_p','mean_r','mean_f1']
retina_columns = ['0_p','0_r','0_f1','1_p','1_r','1_f1','2_p','2_r','2_f1','3_p','3_r','3_f1','4_p','4_r','4_f1','mean_p','mean_r','mean_f1']
method2path = {
	'CE' : ['run/result/xray3image_result/iterNo{}/128d-softmax_cf'.format(i+1) for i in range(5)],
	'WCE' : ['run/result/xray3image_result/iterNo{}/128d-CE-no_class_weights'.format(i+1) for i in range(5)],
	'OCE' : ['run/result/xray3image_result/iterNo{}/128d-oversampling-softmax_cf'.format(i+1) for i in range(5)],
	'Focal' : ['run/result/xray3image_result/iterNo{}/128d-focalloss'.format(i+1) for i in range(5)],
	'W-Focal' : ['run/result/xray3image_result/iterNo{}/128d-weight-focalloss'.format(i+1) for i in range(5)],
	'Triplet' : ['run/result/xray3image_result/iterNo{}/margin0.5_128d-embedding_RandomNegativeTripletSelector'.format(i+1) for i in range(5)],
	'T-Center' : ['run/xray3/iter{}/Centerloss_margin0.5_128d'.format(i+1) for i in range(5)]
} 


if __name__ == '__main__':
	Result = []
	for method in method2path:
		index = ['{}|iter{}'.format(method,i) for i in range(5)]
		columns = xray13_columns
		fileList = [os.path.join(iterName,'result.csv') for iterName in method2path[method]]
		dfList = [pd.read_csv(file) for file in fileList]
		resList = [df[-20:].to_numpy().mean(axis=0) for df in dfList]
		methodStat = pd.DataFrame(resList,index=index,columns=columns)
		methodStat = methodStat.append(
			pd.DataFrame([methodStat.mean().to_numpy()],
				index=['{}|mean'.format(method)],
				columns=xray13_columns)
			)
		Result.append(methodStat)
	Result = pd.concat(Result)
	Result.to_csv('stat_20.csv')




