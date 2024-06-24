library(mgcv)
library(simcausal)
library(fairmodels)
library(DALEX)
library(fairadapt)
set.seed(940984) # 940987
# setwd("C:/Users/ua341au/Dropbox/Documents/GitHub/paper_2023_causal_fairness")
source("code/func_help.R")

#----------------------------------#
#### Structure of this code ########
#----------------------------------#
# 1. Learn warping models
# 2. Train models in both worlds
# 3. Apply warping on test data
# 4. Predict in warped world
# 5. Evaluation
# 6. Classical fairML metrics
#----------------------------------#

save_plots <- TRUE
height <- 5

#----------------------------------#
#### 0. Preparation data ####
#----------------------------------#

a <- Sys.time()

# Load preprocessed german credit data, see "code/german_credit_prepro.R" for details
german_credit <- readRDS(file="data/german_credit_preprocessed.Rda")
dat_g <- german_credit[, c("Sex", "Age", "Saving.accounts", "Credit.amount", "Risk")]
colnames(dat_g) <-  c("Sex", "Age", "Saving", "Amount", "Risk")
dat_g$Saving <- as.numeric(dat_g$Saving == "little")
cols <- c("Age", "Amount", "Saving")

# Train-Test split
# TODO: Sollten wir hier CV oder so machen?
train_ids <- sample(seq_len(nrow(dat_g)), size = 800, replace = FALSE)
dat_train <- dat_g[train_ids,] 
dat_test <- dat_g[-train_ids,]

dim(dat_train)
dim(dat_test)

# gender-rows 
male_rows <- dat_train$Sex == "male"
female_rows <- dat_train$Sex != "male"

# Sub-data sets for each gender
german_m <- dat_train[male_rows,]
german_f <- dat_train[female_rows,]

#----------------------------------#
#### 1. Learn warping models and warp training data ####
#----------------------------------#

### Amount
# Male and female model for warping
g_mod_am_f <- glm(Amount ~ 1 + Age, family = Gamma(link="log"), data = german_f)
g_mod_am_m <- glm(Amount ~ 1 + Age, family = Gamma(link="log"), data = german_m)
# summary(g_mod_am_f)
# summary(g_mod_am_m)

# Warp female amount values to male model
g_am_star_f <- warp_model(dat_x = german_f, 
                          model_x = g_mod_am_f, 
                          model_y = g_mod_am_m)

### Saving
g_mod_sav_f <- glm(Saving ~ 1 + Age, family = binomial, data = german_f)
g_mod_sav_m <- glm(Saving ~ 1 + Age, family = binomial, data = german_m)
# summary(g_mod_sav_f)
# summary(g_mod_sav_m)

g_sav_star_f <- warp_model(dat_x = german_f, 
                           model_x = g_mod_sav_f, 
                           model_y = g_mod_sav_m)

### Risk
g_mod_risk_f <- glm(Risk ~ 1 + Age + Saving + Amount, 
                    family = binomial, data = german_f)
g_mod_risk_m <- glm(Risk ~ 1 + Age + Saving + Amount, 
                    family = binomial, data = german_m)
# summary(g_mod_risk_f)
# summary(g_mod_risk_m)

german_f_warped <- german_f
german_f_warped[,"Saving"] <- g_sav_star_f
german_f_warped[,"Amount"] <- g_am_star_f

g_risk_star_f <- warp_model(dat_x = german_f_warped, 
                            model_x = g_mod_risk_f, 
                            model_y = g_mod_risk_m)

# german_f_warped[,"Risk"] <- factor(round(g_risk_star_f), levels = c(0,1),
#                                    labels=c("bad", "good"))

german_f_warped[,"Risk"] <- round_risk(g_risk_star_f,
                                    as.numeric(german_m$Risk)-1)

# all.equal(german_f_warped[,"Risk_old"], german_f_warped[,"Risk"])
german_warped <- rbind(german_m, german_f_warped)
german_warped <- german_warped[order(as.numeric(rownames(german_warped))),]
head(german_warped, 20)
head(dat_train[order(as.numeric(rownames(dat_train))),], 20)


#----------------------------------#
#### 2. Train models in both worlds ####
#----------------------------------#

g_mod_real <- glm(Risk ~ ., family = binomial, data = dat_train)
g_mod_real_wo_sex <- glm(Risk ~ . - Sex, family = binomial, data = dat_train)
g_mod_warped <- glm(Risk ~ ., family = binomial, data = german_warped)
g_mod_warped_wo_sex <- glm(Risk ~ . - Sex, family = binomial, data = german_warped)

summary(g_mod_real)
summary(g_mod_real_wo_sex)
summary(g_mod_warped)
summary(g_mod_warped_wo_sex)



#----------------------------------#
# 3. Warp test data with training data warping model ####
#----------------------------------#

dat_test_real_female <- dat_test_warped_female <- dat_test[dat_test$Sex=="female",]
dat_test_real_male <- dat_test[dat_test$Sex=="male",]

# Warp female "Amount" values to male model
dat_test_warped_female[,"Amount"] <- warp_new_data(dat_new = dat_test_warped_female, 
                                                   model_x = g_mod_am_f, 
                                                   model_y = g_mod_am_m,
                                                   target = "Amount")

# Warp female "Saving" values to male model
dat_test_warped_female[,"Saving"] <- warp_new_data(dat_new = dat_test_warped_female, 
                                                   model_x = g_mod_sav_f, 
                                                   model_y = g_mod_sav_m,
                                                   target = "Saving")

# Warp female "Risk" values to male model
# str(dat_test_warped_female[,"Risk"])
dat_test_warped_female[,"Risk"] <- as.numeric(dat_test_warped_female[,"Risk"]) - 1
# str(dat_test_warped_female[,"Risk"])
dat_test_warped_female[,"Risk"] <- warp_new_data(dat_new = dat_test_warped_female, 
                                                 model_x = g_mod_risk_f, 
                                                 model_y = g_mod_risk_m,
                                                 target = "Risk")

end_warp <- Sys.time()

#----------------------------------#
#### 7. fairadapt ####
#----------------------------------#


# initialising the adjacency matrix
# adj.mat <- c(
#   0, 0, 1, 1, 1, # Gender
#   0, 0, 1, 1, 1, # Age
#   0, 0, 0, 0, 1, # Amount
#   0, 0, 0, 0, 1, # Saving
#   0, 0, 0, 0, 0  # Risk
# )
adj.mat <- c(
  0, 0, 1, 1, 1, 0, # Sex
  0, 0, 1, 1, 1, 0, # Age
  0, 0, 0, 0, 1, 0, # Saving
  0, 0, 0, 0, 1, 0, # Amount
  0, 0, 0, 0, 0, 1, # Risk
  0, 0, 0, 0, 0, 0  # pseudo-target Z
)

#vars <- c("Sex", cols, "Risk")
vars <- c("Sex", cols, "Risk", "Z")
adj.mat <- matrix(adj.mat, 
                  nrow = length(vars), 
                  ncol = length(vars),
                  dimnames = list(vars, vars), 
                  byrow = TRUE)


# dat_test$Risk <- factor(dat_test$Risk, 
#                               levels = c(0:1), 
#                               labels = c("bad", "good"))

dat_train$Z <- rnorm(nrow(dat_train))
dat_test$Z <- rnorm(nrow(dat_test))

dat_train$Sex <- relevel(dat_train$Sex, ref = "male")
dat_test$Sex <- relevel(dat_test$Sex, ref = "male")

mod <- fairadapt(Z ~ ., # Z
                 train.data = dat_train,
                 test.data = dat_test, 
                 prot.attr = "Sex", 
                 adj.mat = adj.mat,
                 visualize.graph=TRUE#, 
                 #res.vars = "hours_per_week"
)

adapt.train <- adaptedData(mod)
adapt.test  <- adaptedData(mod, train = FALSE)
summary(mod)
summary(adapt.train)
summary(adapt.test)

# adapt_train <- adapt.train
adapt_train <- adapt.train[,vars[1:5]]
adapt_train$Sex <- dat_train$Sex
#adapt_test <- adapt.test
adapt_test <- adapt.test[,vars[1:5]]
adapt_test$Sex <- dat_test$Sex
summary(adapt_train)
summary(adapt_test)

adapt_train$Risk <- relevel(adapt_train$Risk, ref = "bad")
adapt_train$Sex <- relevel(adapt_train$Sex, ref = "female")

adapt_test$Risk <- relevel(adapt_test$Risk, ref = "bad")
adapt_test$Sex <- relevel(adapt_test$Sex, ref = "female")

g_mod_adapt <- glm(Risk ~., family = binomial, 
                   data = adapt_train)
summary(g_mod_adapt)

end_adapt <- Sys.time()

#----------------------------------#
#### 4. Predict in warped world ####
#----------------------------------#

# Predict target for test data

pred_female_warped <- predict(g_mod_warped, newdata = dat_test_warped_female, type="response")
pred_male_warped <- predict(g_mod_warped, newdata = dat_test_real_male, type="response")

pred_female_real <- predict(g_mod_real, newdata = dat_test_real_female, type="response")
pred_male_real <- predict(g_mod_real, newdata = dat_test_real_male, type="response")

pred_female_adapt <- predict(g_mod_adapt, newdata = adapt_test[adapt_test$Sex=="female",], type="response")
pred_male_adapt <- predict(g_mod_adapt, newdata = adapt_test[adapt_test$Sex=="male",], type="response")

b <- Sys.time()

print(b-a)
print(end_warp-a)
print(end_adapt-end_warp)
print(b-end_adapt)

#----------------------------------#
#### 5. Evaluation ####
#----------------------------------#

#--------------#
# 1) Test performance
# (UC1)

mean(round(pred_female_real) == (as.numeric(dat_test_real_female$Risk)-1))
mean(round(pred_male_real) == (as.numeric(dat_test_real_male$Risk)-1))

mean(round(pred_female_warped) == round(dat_test_warped_female$Risk))
mean(round(pred_male_warped) == (as.numeric(dat_test_real_male$Risk)-1))

# => Test performance for females is slightly better in warped world

# This is not entirely ok: Apparently, fairadapt does not warp the target
#   on test
# Done: Find out why fairadapt does not warp target on test
#mean(round(pred_female_adapt) == (as.numeric(dat_test_real_female$Risk)-1))
mean(round(pred_female_adapt) == (as.numeric(adapt_test[adapt_test$Sex=="female",]$Risk)-1))
mean(round(pred_male_adapt) == (as.numeric(dat_test_real_male$Risk)-1))

# # fair comparison would hence be
# mean(round(pred_female_warped) == (as.numeric(dat_test_real_female$Risk)-1))
# mean(round(pred_male_warped) == (as.numeric(dat_test_real_male$Risk)-1))

mean(round(pred_female_adapt) == round(dat_test_warped_female$Risk))


#--------------#
# 2) Mapping
#   a) Features: Which features vary most between the 2 worlds?
# (UC4)

# MSEs of normalized values
mse_vec <- mse_func_col(dat_test_warped_female, dat_test_real_female, cols=cols)
print(mse_vec)

mse_vec_adapt_female <- mse_func_col(adapt_test[adapt_test$Sex=="female",], 
                                     dat_test_real_female, cols=cols)
print(mse_vec_adapt_female)

mse_vec_adapt_male <- mse_func_col(adapt_test[adapt_test$Sex=="male",], 
                                   dat_test_real_male, cols=cols)
print(mse_vec_adapt_male)
# => Apparently, fairadapt warped male to female..
# Can we force the other way around?
# Fixed: with relevel()

#----------#
#   b) Observations: Which observations change the most between the 2 worlds?
# (UC3)

dat_eval_map <- dat_test_real_female[,cols]
dat_eval_map$a_warped <- dat_test_warped_female[,"Amount"]
dat_eval_map$s_warped <- dat_test_warped_female[,"Saving"]
dat_eval_map$mse_row_wr <- mse_func_row(dat_test_warped_female, 
                                        dat_test_real_female, cols=cols)
dat_eval_map[,-1] <- round(dat_eval_map[,-1], 3)
head(dat_eval_map[order(dat_eval_map$mse, decreasing=TRUE),])

mod_warp_change_sim <- gam(mse_row_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_map)
summary(mod_warp_change_sim)
plot(mod_warp_change_sim, pages=1)


plot(mod_warp_change_sim, select=1, main = "partial effect of age on warping distance")

# => Young women are warped most strongly

 
dat_eval_map_adapt <- dat_test_real_female[,cols]
dat_eval_map_adapt$a_warped <- adapt_test[adapt_test$Sex=="female","Amount"]
dat_eval_map_adapt$s_warped <- adapt_test[adapt_test$Sex=="female","Saving"]
dat_eval_map_adapt$mse_row_wr <- mse_func_row(adapt_test[adapt_test$Sex=="female",], 
                                              dat_test_real_male, cols=cols)
dat_eval_map_adapt[,-1] <- round(dat_eval_map_adapt[,-1], 3)
head(dat_eval_map_adapt[order(dat_eval_map_adapt$mse, decreasing=TRUE),])

mod_warp_change_sim_adapt <- gam(mse_row_wr~s(Age) + s(Amount) + Saving, 
                                 dat=dat_eval_map_adapt)
summary(mod_warp_change_sim_adapt)
plot(mod_warp_change_sim_adapt, pages=1)

plot(mod_warp_change_sim_adapt, select=1, main = "partial effect of age on warping distance")

# => Young and old women are warped most strongly with fairadapt

#------------#
# 3) ML model
#   b) Compare predictions in the 2 worlds => similar to 2b)
# (UC2)

dat_eval_pred <- dat_test_real_female[,cols]
dat_eval_pred$a_warped <- dat_test_warped_female[,"Amount"]
dat_eval_pred$s_warped <- dat_test_warped_female[,"Saving"]
dat_eval_pred$pf_real <- round(pred_female_real,2)
dat_eval_pred$pf_warped <- round(pred_female_warped,2)
dat_eval_pred$diff_f_wr <- round(pred_female_warped-pred_female_real,2)

head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=TRUE),])
head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=FALSE),])

# Same for male individuals
dat_eval_pred_male <- dat_test_real_male[,cols]
dat_eval_pred_male$pm_real <- round(pred_male_real,2)
dat_eval_pred_male$pm_warped <- round(pred_male_warped,2)
dat_eval_pred_male$diff_m_wr <- round(pred_male_warped-pred_male_real,2)
head(dat_eval_pred_male[order(dat_eval_pred_male$diff_m_wr, decreasing=TRUE),])
head(dat_eval_pred_male[order(dat_eval_pred_male$diff_m_wr, decreasing=FALSE),])

## ## ## ##
# fairadapt
dat_eval_pred_adapt <- dat_test_real_female[,cols]
dat_eval_pred_adapt$a_warped <- adapt_test[adapt_test$Sex=="female","Amount"]
dat_eval_pred_adapt$s_warped <- adapt_test[adapt_test$Sex=="female","Saving"]
dat_eval_pred_adapt$pf_real <- round(pred_female_real,2)
dat_eval_pred_adapt$pf_warped <- round(pred_female_adapt,2)
dat_eval_pred_adapt$diff_f_wr <- round(pred_female_adapt-pred_female_real,2)

head(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing=TRUE),])
head(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing=FALSE),])

# Same for male individuals
dat_eval_pred_male_adapt <- dat_test_real_male[,cols]
dat_eval_pred_male_adapt$pm_real <- round(pred_male_real,2)
dat_eval_pred_male_adapt$pm_warped <- round(pred_male_adapt,2)
dat_eval_pred_male_adapt$diff_m_wr <- round(pred_male_adapt-pred_male_real,2)
head(dat_eval_pred_male_adapt[order(dat_eval_pred_male_adapt$diff_m_wr, decreasing=TRUE),])
head(dat_eval_pred_male_adapt[order(dat_eval_pred_male_adapt$diff_m_wr, decreasing=FALSE),])

## ## ## ##
if(save_plots){
  pdf("plots/german_2b.pdf", width=height, height=height)
}
boxplot(dat_eval_pred$diff_f_wr, dat_eval_pred_male$diff_m_wr, xlab=NULL, 
        main="Prediction difference warped-real", ylab="Prediction difference",
        ylim=c(-0.09,0.25))
axis(1,at=c(1:2) ,labels=c("Female", "Male"), las=1)
dev.off()
mean(dat_eval_pred$diff_f_wr)
mean(dat_eval_pred_male$diff_m_wr)

t.test(dat_eval_pred$diff_f_wr)$p.value
t.test(dat_eval_pred_male$diff_m_wr)$p.value

# => on average, female preds change, male preds do not change.. 

## ## ## ##
# fairadapt
if(save_plots){
  pdf("plots/german_2b_adapt.pdf", width=height, height=height)
}
boxplot(dat_eval_pred_adapt$diff_f_wr, dat_eval_pred_male_adapt$diff_m_wr, xlab=NULL, 
        main="Prediction difference adapt-real", ylab="Prediction difference",
        ylim=c(-0.09,0.25))
axis(1,at=c(1:2) ,labels=c("Female", "Male"), las=1)
dev.off()

mean(dat_eval_pred_adapt$diff_f_wr)
mean(dat_eval_pred_male_adapt$diff_m_wr)

t.test(dat_eval_pred_adapt$diff_f_wr)$p.value
t.test(dat_eval_pred_male_adapt$diff_m_wr)$p.value

## ## ## ##

# Regression of change on features - female
mod_pred_change_female <- gam(diff_f_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_pred)
summary(mod_pred_change_female)
if(save_plots){
  pdf("plots/german_1.pdf", width=height+2, height=height)
}
#plot(mod_pred_change_female, pages=1, main = "Partial effect", ylim=c(-0.1, 0.18), xlim=c(20,60))
par(mfrow=c(1,2))
plot(mod_pred_change_female, select=1, main = "PE Age - Female", ylim=c(-0.1, 0.18), xlim=c(19,65))
plot(mod_pred_change_female, select=2, main = "PE Amount - Female", ylim=c(-0.1, 0.18), xlim=c(0,12000))
#plot(mod_pred_change_female, select=1, main = "Partial effect")
dev.off()

# Regression of change on features - male
mod_pred_change_male <- gam(diff_m_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_pred_male)
summary(mod_pred_change_male)
if(save_plots){
  pdf("plots/german_3.pdf", width=height+2, height=height)
}
par(mfrow=c(1,2))
plot(mod_pred_change_male, select=1, main = "PE Age - Male", ylim=c(-0.1, 0.18), xlim=c(19,65))
plot(mod_pred_change_male, select=2, main = "PE Amount - Male", ylim=c(-0.1, 0.18), xlim=c(0,12000))

dev.off()
# => also for males there is quite a change!

## ## ## ##
# fairadapt
mod_pred_change_female_adapt <- gam(diff_f_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_pred_adapt)
#plot(mod_pred_change_female_adapt, select=1, main = "Partial effect on prediction difference")
plot(mod_pred_change_female_adapt, pages=1, main="Partial effect")
mod_pred_change_male_adapt <- gam(diff_m_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_pred_male_adapt)
#plot(mod_pred_change_male_adapt, select=1, main = "Partial effect on prediction difference")
plot(mod_pred_change_male_adapt, pages=1, main="Partial effect")
## ## ## ##

# Compare ranks: male
pred_order_male_real <- as.numeric(rownames(dat_eval_pred_male[order(dat_eval_pred_male$pm_real, decreasing=TRUE),]))
male_df_ranks_real <- data.frame(ID = pred_order_male_real, rank_real = seq_len(length(pred_order_male_real)))

pred_order_male_warped <- as.numeric(rownames(dat_eval_pred_male[order(dat_eval_pred_male$pm_warped, decreasing=TRUE),]))
male_df_ranks_warped <- data.frame(ID = pred_order_male_warped, rank_warped = seq_len(length(pred_order_male_warped)))

male_df_ranks <- merge(male_df_ranks_warped, male_df_ranks_real, by = "ID")
# barplot(table(male_df_ranks$rank_real-male_df_ranks$rank_warped))
boxplot(dat_eval_pred_male$pm_real, dat_eval_pred_male$pm_warped, xlab=NULL, main="Risk predictions male")
axis(1,at=c(1:2) ,labels=c("male real", "male warped"), las=1)
for (i in seq_len(nrow(male_df_ranks))){
  segments(1, dat_eval_pred_male$pm_real[i], 2, dat_eval_pred_male$pm_warped[i], col = "gray", lty = "solid")
}

# female
pred_order_female_real <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_real, decreasing=TRUE),]))
female_df_ranks_real <- data.frame(ID = pred_order_female_real, rank_real = seq_len(length(pred_order_female_real)))

pred_order_female_warped <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_warped, decreasing=TRUE),]))
female_df_ranks_warped <- data.frame(ID = pred_order_female_warped, rank_warped = seq_len(length(pred_order_female_warped)))

female_df_ranks <- merge(female_df_ranks_warped, female_df_ranks_real, by = "ID")
# barplot(table(female_df_ranks$rank_real-female_df_ranks$rank_warped))

if(save_plots){
  pdf("plots/german_4b.pdf", width=height, height=height)
}
boxplot(dat_eval_pred$pf_real, dat_eval_pred$pf_warped, xlab=NULL, 
        main="RPID - females", ylab="Risk prediction",
        ylim=c(0.35, 0.9))
axis(1,at=c(1:2) ,labels=c("Real world", "Warped world"), las=1)
for (i in seq_len(nrow(dat_eval_pred))){
  segments(1, dat_eval_pred$pf_real[i], 2, dat_eval_pred$pf_warped[i], col = "gray", lty = "solid")
}

for(j in 1:4){
  segments(1, dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing = TRUE),]$pf_real[j], 
         2, dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing = TRUE),]$pf_warped[j], 
         col = "black", lty = "solid")
}
dev.off()

# => the four most warped females changed saving from 1 to 0


## ## ## ##
# fairadapt

# Compare ranks: male
pred_order_male_adapt <- as.numeric(rownames(dat_eval_pred_male_adapt[order(dat_eval_pred_male_adapt$pm_warped, decreasing=TRUE),]))
male_df_ranks_adapt <- data.frame(ID = pred_order_male_adapt, 
                                  rank_warped = seq_len(length(pred_order_male_adapt)))

male_df_ranks_adapt <- merge(male_df_ranks_adapt, male_df_ranks_real, by = "ID")
# barplot(table(male_df_ranks_adapt$rank_real-male_df_ranks_adapt$rank_warped))
boxplot(dat_eval_pred_male_adapt$pm_real, dat_eval_pred_male_adapt$pm_warped, xlab=NULL, main="Risk predictions male")
axis(1,at=c(1:2) ,labels=c("male real", "male adapt"), las=1)
for (i in seq_len(nrow(male_df_ranks_adapt))){
  segments(1, dat_eval_pred_male_adapt$pm_real[i], 2, dat_eval_pred_male_adapt$pm_warped[i], col = "gray", lty = "solid")
}

# female
pred_order_female_adapt <- as.numeric(rownames(dat_eval_pred_adapt[order(dat_eval_pred_adapt$pf_warped, decreasing=TRUE),]))
female_df_ranks_adapt <- data.frame(ID = pred_order_female_adapt, 
                                    rank_warped = seq_len(length(pred_order_female_adapt)))

female_df_ranks_adapt <- merge(female_df_ranks_adapt, female_df_ranks_real, by = "ID")
# barplot(table(female_df_ranks_adapt$rank_real-female_df_ranks$rank_warped))
if(save_plots){
  pdf("plots/german_4b_adapt.pdf", width=height, height=height)
}
boxplot(dat_eval_pred_adapt$pf_real, dat_eval_pred_adapt$pf_warped, xlab=NULL, 
        main="fairadapt - females", ylab="Risk prediction",
        ylim=c(0.35, 0.9))
axis(1,at=c(1:2) ,labels=c("Real world", "Adapt world"), las=1)
for (i in seq_len(nrow(dat_eval_pred_adapt))){
  segments(1, dat_eval_pred_adapt$pf_real[i], 2, dat_eval_pred_adapt$pf_warped[i], col = "gray", lty = "solid")
}
for(j in 1:4){
  segments(1, dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing = TRUE),]$pf_real[j], 
           2, dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, decreasing = TRUE),]$pf_warped[j], 
           col = "black", lty = "solid")
}
dev.off()
## ## ## ##

plot(density(dat_test_real_female$Age), lwd=2)
lines(density(dat_train$Age[dat_train$Sex=="male"]), col="blue", lty=2, lwd=2)
legend("topright", legend=c("male", "female"), col = c("blue", "black"), lty = c(1:2), lwd=2)
# => Well, female applicants seem to be younger..
# => Should we warp age, too?

# RPID female
xtable(head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, 
                                decreasing=TRUE),]),
       digits=c(0,0,0,0,0,0,2,2,2))
xtable(tail(dat_eval_pred[order(dat_eval_pred$diff_f_wr, 
                                 decreasing=TRUE),]),
       digits=c(0,0,0,0,0,0,2,2,2))

# RPID male
xtable(head(dat_eval_pred_male[order(dat_eval_pred_male$diff_m_wr, 
                                     decreasing=TRUE),]),
       digits=c(0,0,0,0,2,2,2))
xtable(tail(dat_eval_pred_male[order(dat_eval_pred_male$diff_m_wr, 
                                    decreasing=TRUE),]),
       digits=c(0,0,0,0,2,2,2))

# Fairadapt female
xtable(head(dat_eval_pred_adapt[order(dat_eval_pred_adapt$diff_f_wr, 
                                decreasing=TRUE),]),
       digits=c(0,0,0,0,0,0,2,2,2))

# Fairadapt male
xtable(tail(dat_eval_pred_male_adapt[order(dat_eval_pred_male_adapt$diff_m_wr, 
                                     decreasing=TRUE),]),
       digits=c(0,0,0,0,2,2,2))

#----------------------------------#
#### 6. Classical fairML metrics ####
#----------------------------------#

head(dat_train)
head(german_warped)

# step 1 - create model(s)  
# 
# g_mod_real <- glm(Risk ~ ., family = binomial, data = dat_train)
# g_mod_warped <- glm(Risk ~ ., family = binomial, data = german_warped)
# 
# mod_real <- glm(Risk~., data = dat_train, family=binomial(link="logit"))
# mod_warped <- glm(Risk~., data = german_warped, family=binomial(link="logit"))

# step 2 - create explainer(s)

# numeric y for explain function
y_numeric_real <- as.numeric(dat_train$Risk) -1
y_numeric_warped <- as.numeric(german_warped$Risk) -1
y_numeric_adapt <- as.numeric(adapt.train$Risk) -1




explainer_real <- explain(g_mod_real, data = dat_train[,-5], y = y_numeric_real)
explainer_warped <- explain(g_mod_warped, data = german_warped[,-5], y = y_numeric_warped)
explainer_adapt <- explain(g_mod_adapt, data = adapt_train[,-1], y = y_numeric_adapt)
explainer_ftu <- explain(g_mod_real_wo_sex, data = dat_train[,-5], y = y_numeric_real)

# step 3 - fairness check  

epsilon <- 0.95
fobject_real <- fairness_check(explainer_real,
                               protected = dat_train$Sex,
                               label="logit real-world",
                               privileged = "male", epsilon=epsilon)

fobject_warped <- fairness_check(explainer_warped,
                                 protected = german_warped$Sex,
                                 label="logit warped-world",
                                 privileged = "male", epsilon=epsilon)

fobject_adapt <- fairness_check(explainer_adapt,
                                protected = adapt_train$Sex,
                                label="logit adapt",
                                privileged = "male", epsilon=epsilon)

fobject_ftu <- fairness_check(explainer_ftu,
                                protected = dat_train$Sex,
                                label="logit ftu",
                                privileged = "male", epsilon=epsilon)

print(fobject_real)
print(fobject_warped)
print(fobject_adapt)
print(fobject_ftu)


if(save_plots){
  pdf("plots/german_fair_real.pdf", width=height, height=height)
}
plot(fobject_real)
dev.off()

if(save_plots){
  pdf("plots/german_fair_warped.pdf", width=height, height=height)
}
plot(fobject_warped)
dev.off()

if(save_plots){
  pdf("plots/german_fair_adapt.pdf", width=height, height=height)
}
plot(fobject_adapt)
dev.off()

if(save_plots){
  pdf("plots/german_fair_ftu.pdf", width=height, height=height)
}
plot(fobject_ftu)
dev.off()



# "Total loss" as computed by print.fairness_object
sum(abs(1-fobject_real$fairness_check_data$score))
sum(abs(1-fobject_warped$fairness_check_data$score))
sum(abs(1-fobject_adapt$fairness_check_data$score))
sum(abs(1-fobject_ftu$fairness_check_data$score))

# Number of "passes" as computed by print.fairness_object
sum(epsilon < fobject_real$fairness_check_data$score & 1/epsilon > fobject_real$fairness_check_data$score)
sum(epsilon < fobject_warped$fairness_check_data$score & 1/epsilon > fobject_warped$fairness_check_data$score)
sum(epsilon < fobject_adapt$fairness_check_data$score & 1/epsilon > fobject_adapt$fairness_check_data$score)
sum(epsilon < fobject_ftu$fairness_check_data$score & 1/epsilon > fobject_ftu$fairness_check_data$score)

# => Model in warped world also in "classical metrics" way fairer than in 
# real world
# => What I do not like: This is on train data, but fairmodels does only allow
# evaluation on train data..
# TODO: In that case, should we at least use all the data (i.e., train a new 
# warping on the entire data set?)
