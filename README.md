### 1.주제 선정 배경
- 미래의 여러 Time Step에 대해서 예측을 진행하는 Multi-Horizon Forecasting은 시계열 예측 분야에서 매우 중요한 문제로 미래에 대한 예측은 유통업, 헬스 케어, 금융 분야에서 중요하게 사용되고 있으며 시계열 예측의 성능 향상은 엄청난 잠재력이 있습니다. <br> 시계열 예측은 미래를 예측하거나 보는 데 도움이 됩니다. 이는 판매량 예측과 같은 비즈니스 결정과, 주가 예측, 재고관리 최적화등 개인투자 관련 결정을 내리는 데 매우 중요합니다.<br>


### 2.연구 목적

- 본 연구의 목적은 시계열 예측 모형 이용하여 가상화폐 이더리움의 가격을 예측을 하며 여러가지 다양한 입력 변수들을 통해 입력되는 여러가지 다양한 변수들이 시계열 예측 성능에 미치는 영향도를 파악하는 것입니다. <br>

- 실험은 Baseline으로 RNN 기반의 LSTM 모델과 시계열 데이터 처리에 우수한 확률적 예측 모형인 DeepAR, Attention 기반 구조의 Temporal Fusion Transformers를 활용하여 실험과정으로 찾은 최적의 하이퍼 파라미터의 조합으로 시계열 모델을 구축하고 성능 평가 지표는 백분율 오류를 기반으로 한 정확도 측정 방법인 SMAPE(Symmetric mean absolute percentage error)를 사용하였습니다. <br> 그리고 성능 향상을 위해 암호화폐와 관련된 뉴스 데이터를 이용하여 텍스트 마이닝 기법을 일부 적용하여 비정형 데이터를 공변량으로 활용 하였습니다. 또한 분석에 가치가 있는 항목들에 대한 정보를 수집한후 적절한 통계학적 절차를 통해 필수적인 공변량을 선택하고 딥러닝 시계열 모델에 반영하여 암호 화폐 예측을 위한 공변량의 효과를 파악하고자 했습니다.<br>


### 3.수집 데이터 

- 수집 데이터는 가상화폐의 일자별 종가데이터와, 거래량,  저가, 고가, 변동가, 변동율 등 2017년9월25일 부터 2022년 5월 28일까지 대략 과거 5년 동안의 가상화폐의 거래 지표를 수집하였고 

- 시장 참여자의 투자 심리를 나타낸 공포-탐욕 지수와 산업 변수로 그래픽 카드의 가격을 결정짓는 핵심 부품인 GPU, RAM의 가격이 이더리움 가격 변화를 이끌 것으로 기대하고 GPU 제조 기업NVIDIA와 AMD의 주가, 메모리 제조 기업인 삼성전자, 하이닉스의 주가 데이터를 수집하였습니다.

- 이더리움의 검색 트래픽 시계열 데이터는  구글 트렌드(Google Trends) 지수 데이터를 수집하였고

- 암호화폐는 달러와는 반대로 안전 도피자산으로서 금과 자산 성격이 유사 하다는 점에서 상품시장에서 대표적인 안전자산인 금의 시세를 수집하고, 은, 구리 아연등을 비롯한 원자재 가격을 수집하였습니다.

- 그리고 일본 엔, 유로, 영국 파운드, 스위스 프랑, 캐나다 달러, 스웨덴 크로네 그리고 중국위안, 한국원화의 미 달러환율 데이터와 서부 텍사스유, 두바이유등 유가 데이터를 수집하였습니다.

- 종속 변수인 이더리움 가격 시계열 데이터는 업비트 제공하는 자료를 사용하였습니다.

- 암호화폐 가격의 이동평균 가격과, log변환, 주별, 월별 평균가격을 가공하여 변수를 추가 하였습니다.

- 한국의 공휴일 정보를 원핫인코딩으로 추가하였습니다.

- 그리고 네이버 뉴스 데이터를 수집해 문서 내에 어떤 주제가 들어있고, 주제 간의 비중이 어떤지를 문서 집합 내의 단어 통계를 수학적으로 분석하는 LDA 주제 분류 모형(잠재 디리클레 할당, Latent Dirichlet Allocation )을 이용하였는데 분류된 지표값을 범주화하여 학습데이터에 추가 했습니다.

- 또한 전쟁이나 금리인하와 같은 특정 사건에 대해 화폐 가격이 결정되는 부분이 있기 때문에 수집한 기사에서 korean KeyBERT 모델을 통해 기사의 핵심 키워드를 추출해서 그 부분을 지표화 하여 학습데이터에 추가하였으나 Temporal Fusion Transfomers 모형이 한글 인식을 제대로 못하는 부분이 있어 실질적인 지표로의 활용은 하지 못했습니다.

### 4.시계열 모델<br>
#### 4-1 DeepAR

- 확률적 예측(probabilistic forecasting) 을 하는 모형입니다. DeepAR에는 probabilistic forecasting , Likelihood Model사용, teaching force training, covariate 과 함께 학습하는 등의 몇가지 특징이 있는데<br>

- 단일 예측값이 아닌 해당 시점의 확률 분포를 모델링하는 파라미터가 output으로 출력되어 probabilistic forecasting을 합니다. 이는 불확실성을 고려한 최적의 의사결정 시스템을 구축하는 장점이 있는 모델입니다.<br>

- 일반 RNN과의 가장 큰 차이중 하나는 Likelihood Model을 사용한다는 것인데 데이터 셋 중 0~1 사이의 확률값으로 나오는 실제 데이터의 분포를 찾기 위해 Gaussian Likelihood와 Negative binomial distribution를 사용하여 정확도를 높여줍니다. Negative binomial distribution는 compound Poisson distribution 중 하나로 시간당 구매건수에서 건당 구매목록이 random할때 특히 그 값이 Logarithmic Distribution를 따를때 정확도를 높여줍니다.<br>

- 보통 RNN은 이전 스텝에서의 출력 값을 현재 스텝의 입력값으로 사용하는데 DeepAR은 정확한 데이터로 훈련하기 위해 예측값을 다음 스텝으로 넘기는 것이 아니라 실제값을 매번 입력값으로 사용하는 교사강요(teaching force) 방식으로 훈련을 합니다. 교사 강요 방식으로 훈련하지만 테스트 단계에서는 일반적인 RNN 방식으로 예측합니다.<br>

- covariate 과 함께 학습하기에 시계열 패턴뿐만 아니라 블랙프라이데이에 첫 수요가 발생했다면 다음 블랙프라이데이에도 수요가 발생할 것이라 예측하고, 해당 품목의 재고를 준비할 수 있습니다.<br>

- T길이의 window를 설정해서 학습시키는데 랜덤한 시작점을 갖는 window는 mini batch 개념과 연관이 있습니다. 첫 주문, 첫 수요량 발생하는 시계열 형태를 학습시키기 위해 수요량이 없는 경우도 zero padding을 통해 window에 포함시킵니다.<br>

- DeepAR 모델의 기타 장점으로는 t시점의 결측치를 이전 시점의 관측치로 예측한 분포에서 샘플링함으로써 결측치를 쉽게 대체할 수 있고 또한 binary data의 경우 Bernoulli likelihood를 사용하는등의 다양한 통계적 특성을 가진 데이터에 확장하여 적용이 가능합니다.<br>

- 제품의 수요를 예측할 때 발생하는 cold-start 문제를 비슷한 제품의 수요 데이터를 활용하여 해결합니다.<br>

#### 4-2 Temporal Fusion Transformers

- Temporal Fusion Transformers는 시계열 예측모형에서 SOTA 알고리즘으로 시간에 따라 달라지지만, <br>현재에도 그 값을 알 수 있으며 미래에도 그 값을 알 수 있는 Time Vary Known Input(week, weekofyear, holiday, quarter) 과 <br> 현재에는 관측을 통해서 그 값을 알 수 있지만 미래의 값은 알 수 없는 Observed Input과 함께 학습시키면 <br> 다중 시점에 대한 예측(Multi-Horizon Forecasting) 성능을 높일 수 있다는 개념을 제시하고 static Variable(시간과 관계 없이 변하지 않는 정적 공변량, ex 상점의 위치등), Encoder Variable 그리고 Decoder Variable 간의 Feature Importance 제공하여 모델의 해석력을 확보할 수 있다는 장점이 있는 모델 입니다. <br> 

- Temporal Fusion Transformers의 주요 기능에는 Gating Mechanism, Variable Selection Network, Static Covariate Encoder, Temporal Processing, Prediction Intervals 가 있습니다.<br>

- Gating Mechanism은 불필요한 성분을 스킵하여 광범위한 데이터셋에 대하여 깊이와 복잡성을 조절 가능하게 합니다.<br> 외인성 입력(Exogenous Input) 과 Target 사이의 관계를 예측하는것은 매우 어려운일인데 이를 위해 TFT는 필요할 때만 비선형 처리를 하여 모델의 유연성을 높이고자 GRN(Gated Residual Network)를 사용하였습니다. <br> GRN(Gated Residual Network)는 Input 값과 선택적 context vector(static covariate)를 받아 처리하고<br> ELU activation function(Exponential Linear Unit Activation Function : ELU는 입력이 0보다 크면 Identify Function으로 동작하며, 0보다 작을 때는 일정한 출력) 을 거친 후 <br> GLU(Gated Linear Units)을 사용하여 주어진 데이터셋에서 필요하지 않은 부분을 제거합니다.

- 그 다음으로 Variable Selection Network를 통해 관련 있는 Input Variable만 선택 합니다. <br> 다양한 변수들이 사용가능한 반면, 출력에 대해서 그들의 연관성이나 명확한 기여도는 일반적으로 알기 어려운데 TFT는 Static Covariate와 Time Dependent Covariate 모두에 대하여 예측에 직결되는 성분을 골라내는 것은 물론, 불필요한 noisy 입력들을 줄일 수 있습니다. 이는 예측과 관련없는 값들의 입력을 줄여 성능개선을 할 수 있습니다. <br>

- Temporal Processing을 통해 Observed Input과 Known Input 모두에 대해 장기 단기 시간 관계를 학습합니다. Interpretable Multi-Head Attention을 통해 장기 의존성 및 해석력을 강화하는데 기존의 Multi-Head Attention 구조와는 조금 다르게 각 Head의 값을 공유하기 위해 수정된 Multi-Head Attention 구조를 사용했고 Decoder Masking을 Multi-Head Attention에 적용했습니다.<br>

- 마지막으로 Prediction Intervals을 통해 Quantile을 이용하여 매 Prediction Horizon에 대하여 Target이 존재할 수 있는 범위를 제공합니다. 이때 사용되는 Quantile Loss Function은 값을 최소화하는 방향으로 학습을 진행됩니다.<br>

### 5. 실험 시나리오

- 실험 시나리오는 수집된 데이터를 몇가지 유형의 데이터셋을 만들고 유형별 데이터셋을 통해 성능의 영향도를 확인하고자 하였는데 <br> 

- 첫번째로 수집된 데이터를 있는 그대로 대부분 활용하는것과 <br>

- 여러 변수들중 서로 상관성이 높은 변수들의 선형결합으로 이루어진 주성분이라는 새로운 변수를 만들어 변수들을 요약하고 축소하는 PCA 기법을 활용하여 차원을 줄였는데 이때 Scree plot을 통해 분산 설명력이 80이 넘는 되는 주성분 2개를 할당하고 시계열 모델 input에 필요한 주, 월 등의 범주지표를 합쳐 학습데이터를 만들었습니다. <br>

- 마지막으로 Temporal Fusion Transfomers에서 입력되는 time_varying_unknown_reals 값의 유형에 따라 성능이 편차가 큰것을 확인하고 기존에 수집한 데이터를 F-통계량과 AIC 값을 활용하여 단계적 변수 선택법을 통해 유의미한 독립 변수만을 선택한 학습데이터를 생성하였습니다. <br>

- 이렇게 생성한 3가지 데이터 유형으로 DeepAR과 Temporal Fusion Transformers 모형에 추가하여 성능을 측정해 보았습니다.

- 그리고 optuna를 통해 하이퍼파라미터를 튜닝하였는데 좀 더 효율적으로 하이퍼파라미터를 탐색하기 위해 TPE (Tree-structured Parzen Estimator) 알고리즘을 사용하였습니다. <br> 격자 탐색 방법인 Grid Search와 랜덤 샘플링인 Random Search와 달리 TPE (Tree-structured Parzen Estimator) 알고리즘은 미지의 목적 함수의 형태에 대한 확률적인 추정을 수행하는 Surrogate Model 모델과 <br> 목적 함수에 대한 현재까지의 확률적 추정 결과를 바탕으로 최적 입력값 x 를 찾는 데 있어 가장 유용할 만한 다음 입력값 후보을 추천해 주는 함수인 Acquisition Function 이 존재하여 <br> 미지의 목적 함수(objective function) f(x)를 최대로 만드는 최적해 x 를 찾는 Bayesian Optimization 방법입니다.


### 6. 실험 결론 

- 실험에서는 PCA 차원축소 또는 변수 선택법을 통해 변수를 축소하여 사용하는것보다 다양한 input 데이터를 직접 사용하는것이 DeepAR, Temporal Fusion Transformations 둘다 성능이 비교적 좋았으며 baseline 모델인 LSTM의 경우 예측한 Step의 값을 Future timestamp에 재귀적으로 공급하는 iterated 방법으로 프로그램을 설계하였으나 test 평가에서 실제값이 반영되어 예측의 오류가 발생하여 비교가 어려웠으나, 예측값을 다음 스텝에 학습데이터로 프로그램 설계시 예측결과는 DeepAR, TFT가 Baseline에 비하여 비교적 우수한 성능을 보여주는것을 확인하였습니다.

- 또한 Temporal Fusion Transformations의 Feature importance를 통해 변수들의 중요도를 파악한 결과 <br> 미래 시점의 값을 현재 시점에서도 알 수 있는 변수(time_varying_known_categoricals)에 등록한 한국의 공휴일, 요일, 월 등이 예측성능에 영향을 주고 있는것을 확인 했습니다.

- DeepAR의 경우 PytorchForecasting에서 제공하는 GroupNormalizer 함수를 사용하였으나 성능에 진전을 보이지 않아 MinMaxScaler를 사용하여 보다 낳은 성능이 나타나는것을 확인하였습니다.

- Temporal Fusion Transformations의 특정피처들을 time_varying_unknown_reals로 등록하면 예측을 전혀 못하고 0값을 출력하거나 또는 predict에 대한 오차가 커지는 현상을 발견 하였습니다. <br> 주로 scale이 큰값이 들어오는 경우 predict에 대한 오차가 점점 커지는 현상이 나타났는데 변수들간의 다중공선성과 같은 상관관계가 존재하는것으로 보이며 그 부분에 대한 원인을 명확히 찾지 못하고 성능에 도움이 될것으로 예상되는 feature 리스트를 만들어 수작업으로 여러 케이스의 SMAPE를 관찰하여 실험을 진행하는 어려움이 있었습니다. <br> EDA단계에서 모델에 적합한 feature들을 찾아내는것이 성능향상에 많은 영향을 줄것으로 예상됩니다.


### 7. 아쉬운점

- 국내총생산, 물가, 실업률등 거시경제변수와 금리, s&p500지수, 미국의 신용등급, 유럽발 금융시장 위기 등 암호화폐 가격에 직간접적으로 영향을 미칠 더욱 다양한 변수들에 대한 테스트를 하지 못한것과  텍스트 마이닝을 이용하여 전쟁, 금리인하등의 뉴스 기사의 특정 키워드를 조합해 time_varying_known_categoricals에 사용하지 못한 부분이 다소 아쉬운 부분이였으며 시계열 예측에 도움을 줄수 있는 피쳐들에 대한 EDA를 정교하게 수행하지 못한 부분도 다소 아쉬운 부분이였습니다.


### 8. 참고논문

- Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting (Bryan Lim, Sercan O. Arik, Nicolas Loeff, Tomas Pfister)
- DeepAR: Probabilistic forecasting with autoregressive recurrent networks (D Salinas, V Flunkert, J Gasthaus)
- Attention Is All You Need


