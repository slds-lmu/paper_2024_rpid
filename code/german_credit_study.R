library(mgcv)
library(simcausal)
set.seed(940984)
source("code/func_help.R")

#----------------------------------#
#### Structure of this code ########
#----------------------------------#
# 1. Learn warping models
# 2. Train models in both worlds
# 3. Apply warping on test data
# 4. Predict in warped world
# 5. Evaluation
#----------------------------------#

#----------------------------------#
#### 0. Preparation data ####
#----------------------------------#

# Load preprocessed german credit data
german_credit <- readRDS(file="data/german_credit_preprocessed.Rda")
dat_g <- german_credit[, c("Sex", "Age", "Saving.accounts", "Credit.amount", "Risk")]
colnames(dat_g) <-  c("Sex", "Age", "Saving", "Amount", "Risk")
dat_g$Saving <- as.numeric(dat_g$Saving == "little")
cols <- c("Age", "Amount", "Saving")

# Train-Test split
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


german_f_warped[,"Risk"] <- round_risk(g_risk_star_f,
                                    as.numeric(german_m$Risk)-1)


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
dat_test_warped_female[,"Risk"] <- as.numeric(dat_test_warped_female[,"Risk"]) - 1
dat_test_warped_female[,"Risk"] <- warp_new_data(dat_new = dat_test_warped_female, 
                                                 model_x = g_mod_risk_f, 
                                                 model_y = g_mod_risk_m,
                                                 target = "Risk")

#----------------------------------#
#### 4. Predict in warped world ####
#----------------------------------#

# Predict target for test data

pred_female_warped <- predict(g_mod_warped, newdata = dat_test_warped_female, type="response")
pred_male_warped <- predict(g_mod_warped, newdata = dat_test_real_male, type="response")

pred_female_real <- predict(g_mod_real, newdata = dat_test_real_female, type="response")
pred_male_real <- predict(g_mod_real, newdata = dat_test_real_male, type="response")

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


#--------------#
# 2) Mapping
#   a) Features: Which features vary most between the 2 worlds?
# (UC4)

# MSEs of normalized values
mse_vec <- mse_func_col(dat_test_warped_female, dat_test_real_female, cols=cols)
print(mse_vec)

#----------#
#   b) Observations: Which observations change the most between the 2 worlds?
# (UC3)

dat_eval_map <- dat_test_real_female[,cols]
dat_eval_map$a_warped <- dat_test_warped_female[,"Amount"]
dat_eval_map$s_warped <- dat_test_warped_female[,"Saving"]
dat_eval_map$mse_row_wr <- mse_func_row(dat_test_warped_female, dat_test_real_female, cols=cols)
dat_eval_map[,-1] <- round(dat_eval_map[,-1], 3)
head(dat_eval_map[order(dat_eval_map$mse, decreasing=TRUE),])

mod_warp_change_sim <- gam(mse_row_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_map)
summary(mod_warp_change_sim)
plot(mod_warp_change_sim, pages=1)


plot(mod_warp_change_sim, select=1, main = "partial effect of age on warping distance")

# => Young women are warped most strongly


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


boxplot(dat_eval_pred$diff_f_wr, dat_eval_pred_male$diff_m_wr, xlab=NULL, 
        main="Prediction difference warped-real", ylab="Prediction difference")
axis(1,at=c(1:2) ,labels=c("Female", "Male"), las=1)

mean(dat_eval_pred$diff_f_wr)
mean(dat_eval_pred_male$diff_m_wr)

t.test(dat_eval_pred$diff_f_wr)$p.value
t.test(dat_eval_pred_male$diff_m_wr)$p.value

# => on average, female preds change, male preds do not change.. but:

# Regression of change on features - female
mod_pred_change_female <- gam(diff_f_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_pred)
summary(mod_pred_change_female)

plot(mod_pred_change_female, select=1, main = "Partial effect on prediction difference")

# Regression of change on features - male
mod_pred_change_male <- gam(diff_m_wr~s(Age) + s(Amount) + Saving, dat=dat_eval_pred_male)
summary(mod_pred_change_male)

plot(mod_pred_change_male, pages=1, main="Partial effect")
# => there is quite a change!

# Compare ranks: male
pred_order_male_real <- as.numeric(rownames(dat_eval_pred_male[order(dat_eval_pred_male$pm_real, decreasing=TRUE),]))
male_df_ranks_real <- data.frame(ID = pred_order_male_real, rank_real = seq_len(length(pred_order_male_real)))

pred_order_male_warped <- as.numeric(rownames(dat_eval_pred_male[order(dat_eval_pred_male$pm_warped, decreasing=TRUE),]))
male_df_ranks_warped <- data.frame(ID = pred_order_male_warped, rank_warped = seq_len(length(pred_order_male_warped)))

male_df_ranks <- merge(male_df_ranks_warped, male_df_ranks_real, by = "ID")
barplot(table(male_df_ranks$rank_real-male_df_ranks$rank_warped))
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
barplot(table(female_df_ranks$rank_real-female_df_ranks$rank_warped))

boxplot(dat_eval_pred$pf_real, dat_eval_pred$pf_warped, xlab=NULL, 
        main="Risk predictions females", ylab="Risk prediction")
axis(1,at=c(1:2) ,labels=c("Real world", "Warped world"), las=1)
for (i in seq_len(nrow(dat_eval_pred))){
  segments(1, dat_eval_pred$pf_real[i], 2, dat_eval_pred$pf_warped[i], col = "gray", lty = "solid")
}

# => the four most warped females changed saving from 1 to 0

plot(density(dat_test_real_female$Age), lwd=2)
lines(density(dat_train$Age[dat_train$Sex=="male"]), col="blue", lty=2, lwd=2)
legend("topright", legend=c("male", "female"), col = c("blue", "black"), lty = c(1:2), lwd=2)
# => Female applicants seem to be younger..

