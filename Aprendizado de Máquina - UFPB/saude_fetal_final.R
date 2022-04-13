#Trabalho de Aprendizagem de Maquina
#Saude Fetal

rm(list=ls())

library(readr)
fetal_health <- read_csv("C:/UFPB/Estatística/11º Período/Aprendizagem de Máquina/2º e 3º Notas/fetal_health.csv")
View(fetal_health)

correct_prediction <- function(c){ (c[1]+c[4])/sum(c) }
false_positivo <- function(c){ c[2]/sum(c) }
false_negativo <- function(c){ c[3]/sum(c) }
true_positivo <- function(c){ c[4]/sum(c) }
true_negativo <- function(c){ c[1]/sum(c) }

### tem sim NA, vou apenas omitir.
fetal_health <- na.omit(fetal_health)
dim(fetal_health)
# 2126   22

data_set <- fetal_health

## Transformar dados............................................

# all the data types are numeric
str(data_set)
# fetal_health is categoric but ordinal

# the data here is imbalanced
hist(data_set$fetal_health, main="Fetal Health distribution")
# to lead with this we can do oversampling or undersampling

class_1 = data.frame(data_set[which(data_set$fetal_health == 1), ])
class_2 = data.frame(data_set[which(data_set$fetal_health == 2), ])
class_3 = data.frame(data_set[which(data_set$fetal_health == 3), ])

print('Respective lengths: ')
length(class_1[[1]])
length(class_2[[1]])
length(class_3[[1]])
print('Proportion between classes 1 and 2: ')
nrow(class_1)/nrow(class_2)
print('Proportion between classes 1 and 3: ')
nrow(class_1)/nrow(class_3)


# oversampling

library(dplyr)
# replicating small classes
balanced_c2 = class_2[rep(1:nrow(class_2), 5),]
balanced_c2 = rbind(balanced_c2, sample_n(class_2, 180))
# c2: 1475(add 180)
balanced_c3 = class_3[rep(1:nrow(class_3), 9),]
balanced_c3 = rbind(balanced_c3, sample_n(class_3, 71))
# c3: 1584(add 71)

# concatenating the balanced classes
balanced_data = rbind(balanced_c2, balanced_c3)
new_dataset = rbind(class_1, balanced_data)

# randomizing the whole dataset
set.seed(17012001)
new_dataset = sample(new_dataset)
head(new_dataset)

## Reducao de Dimensionalidade...................................

#Análise de Componentes Principais:
p <- dim(new_dataset[,-12])[2]; p # dimensão de colunas (quantidade de colunas)
R <- cor(new_dataset); R

AF <- eigen(R) #cálculo dos autovetores e autovalores
delta <- AF$values; delta

delta/p # Proporcao da variabilidade
cumsum(delta)/p
# 99% da variabilidade amostral é explicada pelas 13 primeiras variaveis.

pca <- princomp(new_dataset[,-12], cor = T) # Componentes Principais

pca$loadings # Coeficientes das variaveis para cada componente

normalized <- new_dataset[,c(1:14)] # Novo conjunto de dados apos a reducao

# informação sobre nossa variavel de interesse.
##Levels: 0 0.5 1

## Análise Descritiva..........................................
summary(new_dataset)

# target variable
target = new_dataset$fetal_health

# columns
columns = colnames(new_dataset)

# descriptive statistics for each column
statistics <- data.frame(matrix(nrow=15, ncol=8))
statistics[1,1] = "STATISTIC"
statistics[1, 2:8] = c("min","max","mean",
                       "median","range",
                       "stdDeviation","variance")
for(i in 2:15) {
   column = columns[i-1]
   statistics[i,1] = column
   
   occurrencies <- table(new_dataset[[column]])
   sort(occurrencies, decreasing=TRUE)
   mode = occurrencies[1]
   
   statistics[i,2] = min(new_dataset[[column]])
   statistics[i,3] = max(new_dataset[[column]])
   statistics[i,4] = mean(new_dataset[[column]])
   statistics[i,5] = median(new_dataset[[column]])
   statistics[i,6] = max(new_dataset[[column]]) - min(new_dataset[[column]])
   statistics[i,7] = sd(new_dataset[[column]])
   statistics[i,8] = var(new_dataset[[column]])
}

head(statistics)

# correlation matrix (Pearson)
pearson_corr_matrix = data.frame(round(cor(new_dataset), digits=4))

# correlation matrix (Spearman)
spearman_corr_matrix = data.frame(round(cor(new_dataset, method='spearman'), digits=4))

# histograms for each feature
for(i in 1:14) {
   hist(new_dataset[[i]], main=columns[i], col="red")
}


## Normalizacao........................................

# standardization
standardized = scale(new_dataset)
head(standardized)

# normalization
minmax_normalization <- function(x) {
  (x - min(x))/(max(x) - min(x))
}

normalized = as.data.frame(lapply(new_dataset, minmax_normalization))
head(normalized)



## Classificacao...............................................

## Analise Discriminante:

library(caret) #matriz de confusao
library(e1071) #matriz de confusao
library(MASS) #analise discriminante

# Transformar a variavel fetal_health em fator para realizar a classificaÃ§Ã£o:
fator <- factor(normalized$fetal_health, levels = c("0", "0.5", "1"));fator

# Selecionando variaveis que sao boas para discriminar fetal_health (quanto mais distinto for 
# o histograma melhor):
ldahist(data = normalized$accelerations, g = fator)#discrimina bem apenas uma classe
ldahist(data = normalized$fetal_movement, g = fator)#nao discrimina bem
ldahist(data = normalized$uterine_contractions, g = fator)#discrimina bem apenas uma classe
ldahist(data = normalized$light_decelerations, g = fator)#discrimina bem apenas uma classe
ldahist(data = normalized$severe_decelerations, g = fator)#nao discrimina bem
ldahist(data = normalized$prolongued_decelerations, g = fator)
ldahist(data = normalized$abnormal_short_term_variability, g = fator)
ldahist(data = normalized$mean_value_of_short_term_variability, g = fator)
ldahist(data = normalized$percentage_of_time_with_abnormal_long_term_variability, g = fator)
ldahist(data = normalized$histogram_min, g = fator)
ldahist(data = normalized$histogram_number_of_zeroes, g = fator)#discrimina bem apenas uma classe
ldahist(data = normalized$histogram_mode, g = fator)
ldahist(data = normalized$histogram_median, g = fator)

# Para a construcao do discriminate foram usadas apenas vas variaveis que discriminaram
# bem ou discriminaram bem apenas uma classe

#........................................................................

# Modelo por Validacao Cruzada:
ml <- lda(normalized[,-12], normalized$fetal_health, CV = T)
names(ml)

plot(ml)

# Matriz de Confusao:
confusionMatrix(data=ml$class, reference = fator)


cv <- 10
k <- 10

# Matrizes que vao receber as méidas das estimativas de rood mean square error
# a quantidade dessas matrizes 
##############################################################################
cp1 <- matrix(0,cv,k)
cp2 <- matrix(0,cv,k)
cp3 <- matrix(0,cv,k)
cp4 <- matrix(0,cv,k)
cp5 <- matrix(0,cv,k)
cp6 <- matrix(0,cv,k)
cp7 <- matrix(0,cv,k)
cp8 <- matrix(0,cv,k)
cp9 <- matrix(0,cv,k)


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
 for(i in 1:k){
   iteração <- 1
     for(j in 1:cv){ 
       # Separando os 10 folds.
       require(caret); 
       flds1 <- createDataPartition(normalized$fetal_health, times = cv, p = 0.5, list = TRUE)
       ################################################################
       # Lista com os elementos separados para treino. # fold1
       ################################################################
       train1 <- normalized[-flds1[[1]], ]
       train2 <- normalized[-flds1[[2]], ]
       train3 <- normalized[-flds1[[3]], ]
       train4 <- normalized[-flds1[[4]], ]
       train5 <- normalized[-flds1[[5]], ]
       train6 <- normalized[-flds1[[6]], ]
       train7 <- normalized[-flds1[[7]], ]
       train8 <- normalized[-flds1[[8]], ]
       train9 <- normalized[-flds1[[9]], ]
       train10 <- normalized[-flds1[[10]], ]
       mat_treino <- list(train1, train2, train3, train4, train5, train6, train7, train8, train9, train10)
       # Lista com os elementos separados para teste.
       teste1 <- normalized[flds1[[1]], ]
       teste2 <- normalized[flds1[[2]], ]
       teste3 <- normalized[flds1[[3]], ]
       teste4 <- normalized[flds1[[4]], ]
       teste5 <- normalized[flds1[[5]], ]
       teste6 <- normalized[flds1[[6]], ]
       teste7 <- normalized[flds1[[7]], ]
       teste8 <- normalized[flds1[[8]], ]
       teste9 <- normalized[flds1[[9]], ]
       teste10 <- normalized[flds1[[10]], ]
       mat_teste <- list(teste1, teste2, teste3, teste4, teste5, teste6, teste7, teste8, teste9, teste10)       
       
       
# .......................................................................

# Modelo por Ressubstituicao:
discriminate_linear <- lda(mat_treino[[i]]$fetal_health ~ mat_treino[[i]]$accelerations +
                              mat_treino[[i]]$uterine_contractions +
                              mat_treino[[i]]$light_decelerations + 
                              mat_treino[[i]]$prolongued_decelerations +
                              mat_treino[[i]]$abnormal_short_term_variability +
                              mat_treino[[i]]$mean_value_of_short_term_variability +
                              mat_treino[[i]]$percentage_of_time_with_abnormal_long_term_variability +
                              mat_treino[[i]]$histogram_min +
                              mat_treino[[i]]$histogram_number_of_zeroes +
                              mat_treino[[i]]$histogram_mode +
                              mat_treino[[i]]$histogram_median, data = mat_treino[[i]])
# plot(discriminate_linear, xlab =)

# Predicoes:
t1 <- predict(object = discriminate_linear, newdata = mat_teste[[i]][-2483,-12], type = "response")
#predicoes
#d <- 1
#for(d in 1:length(t1$class)){
#  if(t1[d] <= 0.333) t1[d] <- 0
#    if(0.333 < t1[d] && t1[d] <= 0.666) t1[d] <- 0.5
#      else t1[d] <- 1
      
#}
cp1[i,j] <- correct_prediction(table(t1$class,mat_teste[[i]][-2483,12]))



## Regressao Logistica:


# Regressao para fetos com saude Normal:
normal <- relevel(as.factor(mat_treino[[i]][ ,12]), ref = "0")

# logistica_N1 <- glm(normal ~ ., data = normalized[,-12], family = binomial(link = 'logit'))
# summary(logistica_N1)

logistica_N2 <- glm(normal ~ ., data = mat_treino[[i]][,-c(12,9)], family = binomial(link = 'logit'))
# summary(logistica_N2) # Modelo Final

t2 <- predict(logistica_N2, mat_treino[[i]][-2483,-12], type = "response")
#predicoes

cp2[i,j] <- correct_prediction(table(t2,mat_teste[[i]][-2483,12]))

# Regressao para fetos com saude Suspeita:
suspeita <- relevel(as.factor(mat_treino[[i]][ ,12]), ref = "0.5")

# logistica_S1 <- glm(suspeita ~ ., data = normalized[,-12], family = binomial(link = 'logit'))
# summary(logistica_S1)

# logistica_S2 <- glm(suspeita ~ ., data = normalized[,-c(6,12)], family = binomial(link = 'logit'))
# summary(logistica_S2)

# logistica_S3 <- glm(suspeita ~ ., data = normalized[,-c(1,6,12)], family = binomial(link = 'logit'))
# summary(logistica_S3)

# logistica_S4 <- glm(suspeita ~ ., data = normalized[,-c(1,6,10,12)], family = binomial(link = 'logit'))
# summary(logistica_S4)

logistica_S5 <- glm(suspeita ~ ., data = mat_treino[[i]][-2483,-c(1,3,6,10,12)], family = binomial(link = 'logit'))
# summary(logistica_S5) # Modelo Final

t3 <- predict(logistica_S5, mat_treino[[i]][-2483,-12], type = "response")
#predicoes

cp3[i,j] <- correct_prediction(table(t3,mat_teste[[i]][-2483,12]))



# Regressao para fetos com saude Patologica:
patologico <- relevel(as.factor(mat_treino[[i]][-2483,12]), ref = "1")

# logistica_P1 <- glm(patologico ~ ., data = normalized[,-12], family = binomial(link = 'logit'))
# summary(logistica_P1)

# logistica_P2 <- glm(patologico ~ ., data = normalized[,-c(12,14)], family = binomial(link = 'logit'))
# summary(logistica_P2)

logistica_P3 <- glm(patologico ~ ., data = mat_treino[[i]][-2483,-c(12,13,14)], family = binomial(link = 'logit'))
# summary(logistica_P3) # Modelo Final
t4 <- predict(logistica_P3, mat_treino[[i]][-2483,-12], type = "response")

cp4[i,j] <- correct_prediction(table(t4,mat_teste[[i]][-2483,12]))



# Regressao........................................................................

X <- as.matrix(mat_treino[[i]][-2483,-12]) #conjunto de variaveis independentes
Y <- mat_treino[[i]][-2483,12] #variavel dependente

library(glmnet)

set.seed(4536)
x <- model.matrix( mat_treino[[i]][-2483,12] ~ ., data = mat_treino[[i]][-2483, ])[,-12] # # Extract design matrix from the data set except the intercept term
y <- mat_treino[[i]][-2483,12] # Extract only the response variable from the data set

cv_train <- cv.glmnet(x, y, alpha = 1) # 10-fold cross-validation on training set
 
## Regressao Ridge
set.seed(123)

# Validacao cruzada:
model1 <- cv.glmnet(X, Y, alpha = 0, lambda = 10^seq(4, -1, -0.1)) 
# obs: alpha = 0 significa regressao ridge
best_lambda <- model1$lambda.min #melhor lambda

ridge_coeff <- predict(model1, s = best_lambda, type = "coefficients")

t5 <- predict(model1, as.matrix(mat_treino[[i]][-2483,-12]), s = best_lambda)

cp5[i,j] <- correct_prediction(table(t5,mat_teste[[i]][-2483,12]))


## Regressao Elastic Net
#set.seed(4536)
#x <- model.matrix( mat_treino[[i]][-2483,12] ~ ., data = mat_treino[[i]][-2483, ])[,-12] # # Extract design matrix from the data set except the intercept term
#y <- mat_treino[[i]][-2483,12] # Extract only the response variable from the data set
#library(glmnet)
#set.seed(123)

#model2 <- cv.glmnet(x, y, alpha = 0.5, lambda = 10^seq(4, -1, -0.1))
# obs: alpha = 0.5 significa regressao elastic net

# best_lambda2 <- model2$lambda.min
# en_coeff <- predict(model2, s = best_lambda2, type = "coefficients")
# en_coeff #Coeficentes de regressão

#t6 <- predict(model2, as.matrix(mat_treino[[i]][-2483, -12]), s = model2$lambda.min)

#cp6[i,j] <- correct_prediction(table(t6,mat_teste[[i]][-2483,12]))


#k-Vizinhos mais proximos..................................................

library(ISLR)
library(dplyr)

# glimpse(normalized)

#Dividindo os dados em Teste e Treinamento:
library(caTools)

#set.seed(1)
#divisao <- sample.split(mat_treino[[i]]$fetal_health, SplitRatio = 0.75)

#saude_teste <- subset(mat_treino[[i]], divisao == FALSE)
#saude_treinamento <- subset(mat_treino[[i]], divisao == TRUE)

#Aplicando o kNN
# library(class)

#set.seed(1)
#previsoes <- knn(train = as.matrix(mat_treino[[i]][-2483,-12]), test = as.matrix(mat_teste[[i]][-2483,-12]), cl =  as.matrix(mat_treino[-2483,12]), k = 3)

# mean(saude_teste[,12] != previsoes) #taxa de erro (% de classificacao errada)

#Construindo um for para escolher o melhor k:
#previsoes = NULL
#perc.erro = NULL

# for(i in 1:20){
#  set.seed(1)
#t7 <- knn(train = mat_treino[-2483,-12], test = mat_teste[-2483,
 #                                                           12], cl =  mat_teste[-2483,12], k = i)
#  perc.erro <- mean(saude_teste[,12] != previsoes)
#  print(perc.erro)
# }

#   cp7[i,j] <- correct_prediction(table(t7,mat_teste[[i]][-2483,-12]))
   
#Randomly split (50 : 50) the Auto data set into training set and test set.
# For the training set, apply lasso regression with 10-fold cross validation
# and report the best choice for ??. Apply the fitted lasso regression model to
# the test data and report the associated MSE. Also report the lasso estimate 
# of regression coefficients for the test data. Do you see any improvement over
# linear regression due to lasso? Justify your answer.
   
set.seed(4536)
x <- model.matrix( mat_treino[[i]][-2483,12] ~ ., data = mat_treino[[i]][-2483, ])[,-12] # # Extract design matrix from the data set except the intercept term
y <- mat_treino[[i]][-2483,12] # Extract only the response variable from the data set
   
cv_train <- cv.glmnet(x, y, alpha = 1) # 10-fold cross-validation on training set
# plot(cv_train)
   
best_lambda <- cv_train$lambda.min
# message("The value of best lambda is ", best_lambda)
   
# 
library(glmnet)
lambda_grid <- 10^seq(from = 10, to = -2, length = 100)
modelo_lasso <- glmnet(as.matrix(mat_treino[[i]][-2483, -12]), mat_treino[[i]][-2483,12], alpha = 1, lambda = 10 ^ seq (4, -1, -0.1)) # Regressão Ridge
# plot(modelo_lasso)
   
t8 <- predict(modelo_lasso, as.matrix(mat_treino[[i]][-2483, -12]), s = cv_train$lambda.min)
#predicoes

cp8[i,j] <- correct_prediction(table(t8,mat_teste[[i]][-2483,12]))
   

     }
   iteração <- iteração + 1
   print(iteração)
 }
# Para prever todos esses valores é usar o predict.
# fited1 <- predict(m1,banco_teste1, type = "response")

medias <- c(mean(cp1),mean(cp2),
            mean(cp3),mean(cp4),
            mean(cp5),mean(cp6),
            mean(cp7),mean(cp8))
which(medias == max(medias))
which(medias == min(medias))

plot(medias, pch = 20)
# Em geral em todos os bancos o mrlhor método foi com o banco mais desbalanceado com o metodo 
# de regressão logistica em média tendo uma predição correta de 0.9546581 e o pior em média foi 
# analise de discriminante linear com o banco balanceado.


###################################################################
## Olhando o desempenho em geral do modelo de classificação.
###################################################################
a <- c(mean(cp1),
       mean(cp2),
       mean(cp3),
       mean(cp4),
       mean(cp5),
       mean(cp6),
        mean(cp7),
       mean(cp8))

boxplot(a, main = "Desempenho em média dos modelo de classificação", 
        ylab = "Predição correta", xlab = "Modelos de classificação Logit, Probit, ADL, ADQ, Reg Ridge, Lasso, Elastic, Classificação Trees")

# Observando o boxplot é possivel ver o modelo de classificação que tem menor variação sendo nos
# tres cenários os modelo de classificação com melhor desempenho foi mlg com distibuiçao binomial
# com funçao de ligação logit e probit e o modelo de classificação Classificação Trees, que teve a 
# menor varição e em media tem o melhor desempenho. Os modelo de classificação com desempenhos ruins 
# foram os knn com k = 5, 3 e 10, pois a variáçaõ foi alta, tendo classificação
# que tem pedição correta chgeando a 90\%, mas em média fica em torno dos 83\%, porém
# também chega a ter um nívell de predição correta de 65\% apenas, demonstrando
# não ser muito centrado. Porém o pior modelo de classificação de classificação ainda foi o random forest
# tendo o chegando no máximo a 75\% de predição correta.

####################################################################
## Olhando e comparando as variáveis as variáveis simetricas
####################################################################
a1 <- apply(cp1, 2, mean)
a2 <- apply(cp2, 2, mean)
a3 <- apply(cp3, 2, mean)
a5 <- apply(cp5, 2, mean)
a8 <- apply(cp8, 2, mean)


x11()
par(mfrow = c(2,1))
boxplot(a1,a2,a3, a5, a8, main = " ", 
        ylab = "Predição correta", xlab = "Modelos de classificação ADL, Logit, Logit, Reg Ridge, Lasso")
boxplot(a1,b1,c1,d1,h1,i1,j1,k1, main = "Desempenho em média dos modelo de classificação utilizando todas as variáveis", 
        ylab = "Predição correta", xlab = "Modelos de classificação Logit, Probit, ADL, ADQ, Reg Ridge, Lasso, Elastic, Classificação Trees")
# Com todos os métodos e retirando os KNN's é possivel visualizar o desempenho dos melhores modelo de classificação
# tendo um desempenho de 91\% a 95\%, em que os modelo de classificação probit e logit tendo mais outlier encontrando
# a explicação de ter dado como o melhor método, porem em geral não é isso que ocorree, poorque esse
# outlier como é alto e nós procuramos a média alta, sem observar um gráfico não expressa bem a realidade
# do desempenho dos métodos os melhores que tiveram menor variação Analise de Discriminante Linear 
# e quadrático.
par(mfrow = c(1,1))
boxplot(a2,b2,d2,e2,f2,g2,k2, main = "Desempenho em média dos modelo de classificação utilizando as variáveis simétricas", 
        ylab = "Predição correta", xlab = "Modelos de classificação Logit, Probit, ADL, ADQ, KNN5, KNN3, KNN10, Classificação Trees")
par(mfrow = c(2,1))
boxplot(a2,b2,d2,e2,f2,g2,k2, main = "Desempenho em média dos modelo de classificação utilizando as variáveis simétricas", 
        ylab = "Predição correta", xlab = "Modelos de classificação Logit, Probit, ADQ, KNN5, KNN3, KNN10, Classificação Trees")
boxplot(c2,g2,h2,i2,k2, main = "Desempenho em média dos modelo de classificação utilizando as variáveis simétricas", 
        ylab = "Predição correta", xlab = "Modelos de classificação ADL, Reg Ridge, Lasso, Elastic")

# Utilizando os bancos com as componentes principais  simétricas o desempenho dos modelo de classificação foi bem
# melhor variando de 88\% a 94\% de predição correta, incluindo os três modelo de classificação com piores desempenho
# utilizando todas as componentes principais os modelo de classificação de KNN 3, 5 e 10 vizinhos próximos, também
# estão nesse intervalo de predição correta e os que tiveram pior desempenho foram os modelo de classificação de 
# analise de discriminante linear, Regressão Ridge, Lasso e Elasticnet variando de 88\% a 92\%, mesmo
# tendo a variância em todos, melhores modelo de classificação variaram de 90\% a 94\%. 

# O melhor metodo vai ser o que é melhor nessas situações tiver o maior 
# predição correta.

####################################################################
## Olhando o desempenho dos modelo de classificação nos senários 
####################################################################
#
A1 <- c(cp1,cp4)
A2 <- c(cp2,cp5)
A3 <- c(cp3,cp6)
#
A4 <- c(cp7,cp10)
A5 <- c(cp8,cp11)
A6 <- c(cp9,cp12)
#
A7 <- c(cp13,cp16)
A8 <- c(cp14,cp17)
A9 <- c(cp15,cp18)
#
