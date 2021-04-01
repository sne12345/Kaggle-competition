# Kaggle-competition
캐글 대회 참여한 코드입니다. 

# .ipynb 파일이 안 열릴 경우
.py 파일을 열거나 clone해서 확인 부탁드립니다.  
<br />

# XGBoost 파라미터  
▶ n_estimators [default = 100] : 나무의 개수 (=num_boost_round [default = 10] : 파이썬 래퍼에서 적용)

 

▶ early_stopping_rounds 

        최대한 몇 개의 트리를 완성해볼 것인지 

        valid loss에 더이상 진전이 없으면 멈춤

        과적합을 방지할 수 있음, n_estimators 가 높을때 주로 사용.

 

▶ learning_rate [default = 0.1] (=eta [default = 0.3] : 파이썬 래퍼에서 적용)

        학습 단계별로 가중치를 얼만큼 사용할지 결정/ 이전의 결과를 얼마나 반영할건지

        낮은 eta -> 낮은 가중치 -> 다음 단계의 결과물 적게 반영 -> 보수적

        일반적으로 0.01 ~ 0.2

        높은 값으로 다른 파라미터 조절하여 결정한 후, 낮춰서 최적의 파라미 결정

        * gradient boost에서는 기울기의 의미, 작으면 꼼꼼히 내려가고 크면 급하게 내려감

 

▶ min_child_weight [default = 1]

        child 에서 필요한 모든 관측치에 대한 가중치의 최소 합

        이 값보다 샘플 수가 작으면 leaf node가 되는 것

        너무 크면 under-fitting 될 수 있음

        CV로 조절해야함

 

▶ max_depth [default = 6]

        트리의 최대 깊이

        일반적으로 3 ~ 10  

        CV로 조절해야함

 

▶ gamma [default = 0]

        트리에서 추가적으로 가지를 나눌지를 결정할 최소 손실 감소 값

        값이 클수록 과적합 감소 효과

 

▶ subsample [default = 1] (=sub_sample : 파이썬 래퍼에서 적용)

        각 트리마다 데이터 샘플링 비율

        over-fitting 방지

        일반적으로 0.5 ~ 1

 

▶ colsample_bytree [default = 1]

        각 트리마다 feature 샘플링 비율

        일반적으로 0.5 ~ 1

 

▶ reg_lambda [default = 1] (=lambda : 파이썬 래퍼에서 적용)

        L2 regularization(ex. 릿지) 가중치

        클수록 보수적

 

▶ reg_alpha [default = 0] (=alpha : 파이썬 래퍼에서 적용)

        L1 regularization(ex. 라쏘) 가중치

        클수록 보수적

        특성이 매우 많은때 사용해볼만 함

 

▶ scale_pos_weight [default = 1]

        데이터가 불균형할때 사용, 0보다 큰 값

        보통 값을 음성 데이터 수/ 양성 데이터 수 값으로 함

 

 
