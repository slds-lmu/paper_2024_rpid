library(mgcv)
library(simcausal)
library(fairmodels)
library(fairadapt)
library(parallel)
library(DALEX)
set.seed(940984)
# setwd("C:/Users/ua341au/Dropbox/Documents/GitHub/paper_2023_causal_fairness")
source("code/func_help.R")

# Catch arguments from CommandLine
default_args = setNames(list(2, FALSE, FALSE, 4), 
                        c("M_iter", "age_sex", "reverse_warping", "num_cores"))
args = R.utils::commandArgs(trailingOnly = TRUE, 
                            asValues = TRUE, 
                            defaults = default_args)
print(args)

start_time <- Sys.time()

#----------------------------------#
#### Structure of this code ########
#----------------------------------#
# 0. Preparation DAGs
# for each m in 1:M_iter
# 1. Simulate real and find world training data
# 2. "Learn" warping functions
# 3. Simulate real and find world test data
# 4. Warp real world data to warped world -- test data
# 5. Pre-computations for results
# 6. Save data for later use in results_sim.R
#----------------------------------#

num_cores <- args$num_cores
M_iter <- args$M_iter #2 #1000 #number iterations of simulation study
n_train <- 10000 #10000 #100000 #2000 # size of training set
n_test <- 1000 #1000 # size of test set
# prob_male <- 0.69 # I do not know how to plug this into simcausal --> hard-coded for now
# n_male <- n*prob_male
# n_female <- n-n_male
age_sex <- args$age_sex # FALSE # TRUE: Age is simulated as sex-dependent
reverse_warping <- args$reverse_warping #FALSE # TRUE: Warp male to female population
rndseed <- 1425 # 1425 1427
rndseed_test <- 1427
cols <- c("Age", "Saving", "Amount")
epsilon <- 0.95 # Tolerance for classical fairness metrics
date_time <- Sys.time()
date_start <- Sys.Date()
config <-
  data.frame(
    num_cores,
    M_iter,
    n_train,
    n_test,
    age_sex,
    reverse_warping,
    rndseed,
    rndseed_test,
    epsilon,
    date_time
  )


cat(rep("=", 50), "\n", sep = "")
cat("Start simulation with ",
    # M_iter, " iterations, age_sex = ", age_sex,
    # ", reverse warping = ", reverse_warping, ", n_train = ",
    # n_train, ", n_test = ", n_test, ", epsilon = ", epsilon,
    " \n", sep = "")
print(config)
cat(rep("=", 50), "\n", sep = "")


list_real <- list()
list_find <- list()
list_warped <- list()
list_adapt <- list()
list_mse <- list()
list_mse_adapt <- list()
list_map <- list()
list_pred <- list()
list_test_real_male <- list()
list_test_real_female <- list()
if (!reverse_warping) {
  list_test_find_female <- list()
  list_test_warped_female <- list()
  list_test_adapt_female <- list()
} else{
  list_test_find_male <- list()
  list_test_warped_male <- list()
  list_test_adapt_male <- list()
}
fair_check_mat <- matrix(0,
                         nrow = M_iter,
                         ncol = 8,
                         dimnames = list(
                           NULL,
                           c(
                             "loss_real",
                             "loss_warped",
                             "loss_adapt",
                             "loss_find",
                             "n_passes_real",
                             "n_passes_warped",
                             "n_passes_adapt",
                             "n_passes_find"
                           )
                         ))
fair_score_mat <- matrix(0,
                         nrow = M_iter,
                         ncol = 20,
                         dimnames = list(NULL, paste0(
                           c("ACC", "PPV", "FPR", "TPR", "STP"), 
                           c(rep("-real", 5), 
                             rep("-warped", 5), 
                             rep("-adapt", 5), 
                             rep("-find", 5))
                         )))

#----------------------------------#
#### 0. Preparation DAGs ####
#----------------------------------#

#-----------#
# Real World
#-----------#

M <- DAG.empty()
M <- M +
  node("Sex", # Sex
       distr = "rbern", prob = 0.69)+
  node("Age", # Age
       distr = "rgamma", scale = 3.64, shape = 9.76)+ 
  node("Saving", # Saving|Sex, Age,
       distr = "rbern", prob = plogis(4 - 1.25 * Sex - 0.1 * Age))+
  node("Amount", #Amount|Sex, Age,
       distr = "rgamma", scale = 0.74*exp(7.9 + 0.175*Sex + 0.005*Age), shape = 1/0.74)+
  node("Risk", # Risk|Sex, Age, Amount, Saving
       distr = "rbern", prob = plogis(0.9 + 0.1*Age + 1.75*Sex - 0.7*Saving - 0.001*Amount))

# Make age sex-dependent
if(age_sex){
  M <- M + 
    node("Age", # Age
         distr = "rgamma", scale = 1/30*exp(3.5 + 0.25*Sex), shape = 30)
}
plotDAG(set.DAG(M))

Mset <- set.DAG(M)

#-----------#
# FiND World
#-----------#

Mfind_act <- Mset

if(!reverse_warping){
  # Saving, Amount and Risk are now equivalently sampled for both male and female
  #   individuals: replace "Sex" by "1" in those nodes
  find_saving <- node("Saving", # Saving|Age,
                      distr = "rbern", prob = plogis(4 - 1.25 * 1 - 0.1 * Age))
  find_amount <- node("Amount", #Amount|Age,
                      distr = "rgamma", scale = 0.74*exp(7.9 + 0.175 * 1 + 0.005*Age), shape = 1/0.74)
  find_risk <- node("Risk", # Risk|Age, Amount, Saving
                    distr = "rbern", prob = plogis(0.9 + 0.1*Age + 1.75*1 - 0.7*Saving - 0.001*Amount))
}else{
  # Saving, Amount and Risk are now equivalently sampled for both male and female
  #   individuals: replace "Sex" by "0" in those nodes
  find_saving <- node("Saving", # Saving|Age,
                      distr = "rbern", prob = plogis(4 - 1.25 * 0 - 0.1 * Age))
  find_amount <- node("Amount", #Amount|Age,
                      distr = "rgamma", scale = 0.74*exp(7.9 + 0.175 * 0 + 0.005*Age), shape = 1/0.74)
  find_risk <- node("Risk", # Risk|Age, Amount, Saving
                    distr = "rbern", prob = plogis(0.9 + 0.1*Age + 1.75*0 - 0.7*Saving - 0.001*Amount))
  
}

# If age is sex-dependent in real-world: also set "Sex" to "1"
if(age_sex){
  find_age <- node("Age", # Age
                   distr = "rgamma", scale = 1/30*exp(3.5 + 0.25*1), shape = 30)
}

# Stupid intervention, which does not do anything, but I do not know how to 
#   do a NULL intervention for getting "real world" data in the simulation
sex_node <- node("Sex", # Sex
                 distr = "rbern", prob = 0.69)

if(!age_sex){
  # Apply actions
  Mfind_act <- Mfind_act + 
    action("find_world", nodes = c(find_saving, find_amount, find_risk)) +
    action("real_world", nodes = sex_node)
}else{
  Mfind_act <- Mfind_act + 
    action("find_world", nodes = c(find_saving, find_amount, find_risk, find_age)) +
    action("real_world", nodes = sex_node)
}

#-----------#
# Force gender to specific value
#-----------#

# Set Sex to male or female
a_sex_male <- node("Sex", distr = "rbern", prob = 1)
a_sex_female <- node("Sex", distr = "rbern", prob = 0)

if(!age_sex){
  Mfind_act <- Mfind_act + 
    action("real_world_male", nodes = a_sex_male) +
    action("real_world_female", nodes = a_sex_female) +
    action("find_world_male", nodes = c(a_sex_male, find_saving, find_amount, find_risk)) +
    action("find_world_female", nodes = c(a_sex_female, find_saving, find_amount, find_risk))
}else{
  Mfind_act <- Mfind_act + 
    action("real_world_male", nodes = a_sex_male) +
    action("real_world_female", nodes = a_sex_female) +
    action("find_world_male", nodes = c(a_sex_male, find_saving, find_amount, find_risk, find_age)) +
    action("find_world_female", nodes = c(a_sex_female, find_saving, find_amount, find_risk, find_age))
}

#----------------------------------#
#### 1. Simulation real and find world data ####
#----------------------------------#

myfun <- function(m){
  # for(m in seq_len(M_iter)){
  
  if(m<5 | m%%100==0){
    cat("M =", m, "\n")
  }
  
  # Simulate data
  dat_find_act <- simcausal::sim(DAG = Mfind_act, 
                                 actions = c("real_world","find_world"),
                                 n = n_train, 
                                 rndseed = rndseed+m, 
                                 verbose = FALSE)
  
  # Separate both worlds in one data.frame, respectively
  dat_find <- dat_find_act$find_world[,-1]
  dat_find$Age <- round(dat_find$Age)
  dat_find$Sex <- factor(dat_find$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_find$Risk <- factor(dat_find$Risk, levels = c(0:1), labels = c("bad", "good"))
  
  dat_real <- dat_find_act$real_world[,-1]
  dat_real$Age <- round(dat_real$Age)
  dat_real$Sex <- factor(dat_real$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_real$Risk <- factor(dat_real$Risk, levels = c(0:1), labels = c("bad", "good"))
  
  # # check if simulated data are somewhat realistic and in a sensible range
  # summary(dat_real)
  # summary(dat_find)
  
  # gender-rows
  male_rows <- dat_real$Sex == "male"
  female_rows <- dat_real$Sex != "male"
  
  # sub-data sets for each gender
  dat_real_m <- dat_real[male_rows,]
  dat_real_f <- dat_real[female_rows,]
  # dat_find_m <- dat_find[male_rows,]
  # dat_find_f <- dat_find[female_rows,]
  
  #----------------------------------#
  #### 2. Warp real world data to warped world ####
  #----------------------------------#
  
  ### Amount
  # Male and female model for warping
  mod_am_f <- glm(Amount ~ 1 + Age, family = Gamma(link="log"), data = dat_real_f)
  mod_am_m <- glm(Amount ~ 1 + Age, family = Gamma(link="log"), data = dat_real_m)
  # summary(mod_am_f)
  # summary(mod_am_m)
  
  if(!reverse_warping){
    # Warp female amount values to male model
    am_star_f <- warp_model(dat_x = dat_real_f, 
                            model_x = mod_am_f, 
                            model_y = mod_am_m)
  }else{
    # Warp male amount values to female model
    am_star_m <- warp_model(dat_x = dat_real_m, 
                            model_x = mod_am_m, 
                            model_y = mod_am_f)
  }
  
  ### Saving
  mod_sav_f <- glm(Saving ~ 1 + Age, family = binomial, data = dat_real_f)
  mod_sav_m <- glm(Saving ~ 1 + Age, family = binomial, data = dat_real_m)
  # summary(mod_sav_f)
  # summary(mod_sav_m)
  
  if(!reverse_warping){
    sav_star_f <- warp_model(dat_x = dat_real_f, 
                             model_x = mod_sav_f, 
                             model_y = mod_sav_m)
  }else{
    sav_star_m <- warp_model(dat_x = dat_real_m, 
                             model_x = mod_sav_m, 
                             model_y = mod_sav_f)
  }
  
  ### Risk
  mod_risk_f <- glm(Risk ~ 1 + Age + Saving + Amount, family = binomial, data = dat_real_f)
  mod_risk_m <- glm(Risk ~ 1 + Age + Saving + Amount, family = binomial, data = dat_real_m)
  # summary(mod_risk_f)
  # summary(mod_risk_m)
  
  if(!reverse_warping){
    
    dat_f_warped <- dat_real_f
    dat_f_warped[,"Amount"] <- am_star_f  
    dat_f_warped[,"Saving"] <- sav_star_f
    
    risk_star_f <- warp_model(dat_x = dat_f_warped, 
                              model_x = mod_risk_f, 
                              model_y = mod_risk_m)
    
    dat_f_warped[,"Risk"] <- round_risk(risk_star_f, 
                                        as.numeric(dat_real_m$Risk)-1)
    dat_warped <- rbind(dat_real_m, dat_f_warped)
    
  }else{
    dat_m_warped <- dat_real_m
    dat_m_warped[,"Saving"] <- sav_star_m
    dat_m_warped[,"Amount"] <- am_star_m
    
    risk_star_m <- warp_model(dat_x = dat_m_warped, 
                              model_x = mod_risk_m, 
                              model_y = mod_risk_f)
    
    
    dat_m_warped[,"Risk"] <- round_risk(risk_star_m, 
                                        as.numeric(dat_real_f$Risk)-1)
    dat_warped <- rbind(dat_real_f, dat_m_warped)
  }
  
  dat_warped <- dat_warped[order(as.numeric(rownames(dat_warped))),]
  # head(dat_warped)
  # head(dat_real)
  # head(dat_find)
  
  
  
  #----------------------------------#
  #### 2.a) Adapt real world data to adapt world ####
  #----------------------------------#
  
  # Can't do this without test data, I guess?
  # => Move 3. to this place
  
  #----------------------------------#
  # 3. Simulate real and find world test data ####
  #----------------------------------#
  
  dat_test <- simcausal::sim(DAG = Mfind_act, 
                             actions = c("real_world_male", 
                                         "real_world_female",
                                         "find_world_male", # Could be deleted for !reverse_warping bc same as real world
                                         "find_world_female"), # Could be deleted for reverse_warping bc same as real world
                             n = n_test, rndseed = rndseed_test+m, verbose = FALSE)
  
  dat_test_real_male <- dat_test$real_world_male[,-1]
  dat_test_real_female <- dat_test$real_world_female[,-1]
  dat_test_find_male <- dat_test$find_world_male[,-1]
  dat_test_find_female <- dat_test$find_world_female[,-1]
  
  # head(dat_test_real_male)
  # head(dat_test_real_female)
  # head(dat_test_find_male)
  # head(dat_test_find_female)
  
  dat_test_real_male$Sex <- factor(dat_test_real_male$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_test_real_female$Sex <- factor(dat_test_real_female$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_test_find_male$Sex <- factor(dat_test_find_male$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_test_find_female$Sex <- factor(dat_test_find_female$Sex, levels = c(0:1), labels = c("female", "male"))
  
  dat_test_real_male$Age <- round(dat_test_real_male$Age)
  dat_test_real_female$Age <- round(dat_test_real_female$Age)
  dat_test_find_male$Age <- round(dat_test_find_male$Age)
  dat_test_find_female$Age <- round(dat_test_find_female$Age)
  
  if(!reverse_warping){
    # "discr" means the "discriminated PA", i.e., the one 
    #   that differs between real and find world
    dat_test_real_discr <- dat_test_real_female
    dat_test_real_non_d <- dat_test_real_male
  }else{
    dat_test_real_discr <- dat_test_real_male
    dat_test_real_non_d <- dat_test_real_female
  }
  
  #----------------------------------#
  #### 2.a-II) Adapt real world data to adapt world ####
  #----------------------------------#
  
  # initialising the adjacency matrix
  adj.mat <- c(
    0, 0, 1, 1, 1, 0, # Sex
    0, 0, 1, 1, 1, 0, # Age
    0, 0, 0, 0, 1, 0, # Saving
    0, 0, 0, 0, 1, 0, # Amount
    0, 0, 0, 0, 0, 1, # Risk
    0, 0, 0, 0, 0, 0  # pseudo-target Z
  )
  
  vars <- c("Sex", cols, "Risk", "Z")
  adj.mat <- matrix(adj.mat, 
                    nrow = length(vars), 
                    ncol = length(vars),
                    dimnames = list(vars, vars), 
                    byrow = TRUE)
  
  
  
  dat_train_adapt <- dat_real
  dat_test_adapt <- dat_test_real_discr
  dat_test_adapt$Risk <- factor(dat_test_adapt$Risk, 
                                levels = c(0:1), 
                                labels = c("bad", "good"))
  
  dat_train_adapt$Z <- rnorm(nrow(dat_train_adapt))
  dat_test_adapt$Z <- rnorm(nrow(dat_test_adapt))
  
  if(!reverse_warping){
    dat_train_adapt$Sex <- relevel(dat_train_adapt$Sex, ref = "male")
    dat_test_adapt$Sex <- relevel(dat_test_adapt$Sex, ref = "male")
  }else{
    dat_train_adapt$Sex <- relevel(dat_train_adapt$Sex, ref = "female")
    dat_test_adapt$Sex <- relevel(dat_test_adapt$Sex, ref = "female")
  }
  
  mod <- fairadapt(Z ~ ., # Risk
                   train.data = dat_train_adapt,
                   test.data = dat_test_adapt, 
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
  
  adapt_train <- adapt.train[,vars[1:5]]
  adapt_train$Sex <- dat_train_adapt$Sex
  adapt_test <- adapt.test[,vars[1:5]]
  adapt_test$Sex <- dat_test_adapt$Sex
  summary(adapt_train)
  summary(adapt_test)
  
  adapt_train$Risk <- relevel(adapt_train$Risk, ref = "bad")
  adapt_train$Sex <- relevel(adapt_train$Sex, ref = "female")
  
  adapt_test$Risk <- relevel(adapt_test$Risk, ref = "bad")
  adapt_test$Sex <- relevel(adapt_test$Sex, ref = "female")
  
  dat_adapt <- adapt_train[order(as.numeric(rownames(adapt_train))),]
  
  #----------------------------------#
  # Sanity check if sim is as expected
  #----------------------------------#
  
  # Expectation (for !reverse_warping):
  #   [check] (A) age and sex equal in all data sets (exactly, by design)
  #   [check] (B) male_real = male_warped = male_find = male_adapt (exactly, by design)
  # 
  # head(dat_real)
  # head(dat_find)
  # head(dat_warped)
  # head(dat_adapt)
  # # 
  # # # (A)
  # all.equal(dat_find,
  #           dat_real)
  # all.equal(dat_find,
  #           dat_warped)
  # all.equal(dat_find,
  #           dat_adapt)
  # # 
  # # # (B)
  # all.equal(dat_find[dat_find$Sex=="male",],
  #           dat_real[dat_real$Sex=="male",])
  # all.equal(dat_find[dat_find$Sex=="male",],
  #           dat_warped[dat_warped$Sex=="male",])
  # all.equal(dat_find[dat_find$Sex=="male",],
  #           dat_adapt[dat_adapt$Sex=="male",])
  
  
  
  #----------------------------------#
  # Models in different worlds  
  #----------------------------------#
  
  
  mod_real <- glm(Risk ~ ., family = binomial, data = dat_real)
  
  mod_warped <- glm(Risk ~ . - Sex, family = binomial, data = dat_warped)
  mod_warped_sex <- glm(Risk ~ ., family = binomial, data = dat_warped)
  
  mod_find <- glm(Risk ~ . - Sex, family = binomial, data = dat_find)
  mod_find_sex <- glm(Risk ~ ., family = binomial, data = dat_find)
  
  mod_adapt <- glm(Risk ~. - Sex, family = binomial, data = dat_adapt)
  mod_adapt_sex <- glm(Risk ~., family = binomial, data = dat_adapt)
  
  
  if(!reverse_warping){
    # baseline: instead of using warped data, just use male data
    mod_baseline <- glm(Risk ~ Age+Amount+Saving, 
                        family = binomial, 
                        data = dat_real_m)
  }else{
    # baseline: instead of using warped data, just use female data
    mod_baseline <- glm(Risk ~ Age+Amount+Saving, 
                        family = binomial, 
                        data = dat_real_f)
  }  
  
  summary(mod_real)$coef
  # summary(mod_warped)
  summary(mod_warped_sex)$coef
  summary(mod_adapt_sex)$coef
  summary(mod_find)$coef
  # 
  
  #### 2.b) classical fairML metrics ####
  y_numeric_real <- as.numeric(dat_real$Risk) -1
  y_numeric_warped <- as.numeric(dat_warped$Risk) -1
  y_numeric_adapt <- as.numeric(dat_adapt$Risk) -1
  y_numeric_find <- as.numeric(dat_find$Risk) -1
  
  
  explainer_real <- explain(mod_real, data = dat_real[,-5], 
                            y = y_numeric_real, verbose=FALSE)
  explainer_warped <- explain(mod_warped_sex, data = dat_warped[,-5], 
                              y = y_numeric_warped, verbose=FALSE)
  explainer_adapt <- explain(mod_adapt_sex, data = dat_adapt[,-5], 
                             y = y_numeric_adapt, verbose=FALSE)
  explainer_find <- explain(mod_find_sex, data = dat_find[,-5], y = 
                              y_numeric_find, verbose=FALSE)
  # step 3 - fairness check
  
  fobject_real <- fairness_check(explainer_real, verbose=FALSE,
                                 protected = dat_real$Sex,
                                 privileged = "male", epsilon=epsilon)
  
  fobject_warped <- fairness_check(explainer_warped, verbose=FALSE,
                                   protected = dat_warped$Sex,
                                   privileged = "male", epsilon=epsilon)
  
  fobject_adapt <- fairness_check(explainer_adapt, verbose=FALSE,
                                  protected = dat_adapt$Sex,
                                  privileged = "male", epsilon=epsilon)
  
  fobject_find <- fairness_check(explainer_find, verbose=FALSE,
                                 protected = dat_find$Sex,
                                 privileged = "male", epsilon=epsilon)
  # 
  # print(fobject_real)
  # print(fobject_warped)
  # print(fobject_adapt)
  # print(fobject_find)
  # plot(fobject_real)
  # plot(fobject_warped)
  # plot(fobject_adapt)
  plot(fobject_find)
  # "Total loss" as computed by print.fairness_object
  loss_real <- sum(abs(1-fobject_real$fairness_check_data$score))
  loss_warped <- sum(abs(1-fobject_warped$fairness_check_data$score))
  loss_adapt <- sum(abs(1-fobject_adapt$fairness_check_data$score))
  loss_find <- sum(abs(1-fobject_find$fairness_check_data$score))
  
  
  # Number of "passes" as computed by print.fairness_object
  n_passes_real <- sum(epsilon < fobject_real$fairness_check_data$score & 1/epsilon > fobject_real$fairness_check_data$score)
  n_passes_warped <- sum(epsilon < fobject_warped$fairness_check_data$score & 1/epsilon > fobject_warped$fairness_check_data$score)
  n_passes_adapt <- sum(epsilon < fobject_adapt$fairness_check_data$score & 1/epsilon > fobject_adapt$fairness_check_data$score)
  n_passes_find <- sum(epsilon < fobject_find$fairness_check_data$score & 1/epsilon > fobject_find$fairness_check_data$score)
  
  # fair_check_mat[m,] <- c(loss_real, loss_warped, loss_adapt, loss_find,
  #                         n_passes_real, n_passes_warped, n_passes_adapt, n_passes_find)
  
  fair_check_m <- c(loss_real, loss_warped, loss_adapt, loss_find,
                    n_passes_real, n_passes_warped, n_passes_adapt, n_passes_find)
  
  # fair_score_mat[m,] <- c(fobject_real$fairness_check_data$score,
  #                         fobject_warped$fairness_check_data$score,
  #                         fobject_adapt$fairness_check_data$score,
  #                         fobject_find$fairness_check_data$score)
  
  fair_score_m <- c(fobject_real$fairness_check_data$score,
                    fobject_warped$fairness_check_data$score,
                    fobject_adapt$fairness_check_data$score,
                    fobject_find$fairness_check_data$score)
  
  
  # TODO: This is on train because the package is built like this. Should we
  # rather do this on test? (Would not be too hard to compute those metrics
  # from a confusion matrix on test - we do not really need the fancy plot)
  # TODO: At some point, migrate this to mlr3fairness
  
  
  #----------------------------------#
  # 4. Warp test data with training data warping model ####
  #----------------------------------#
  
  if(!reverse_warping){
    # "discr" means the "discriminated PA", i.e., the one 
    #   that differs between real and find world
    dat_test_warped_discr <- dat_test_real_female
    dat_test_real_discr <- dat_test_real_female
    dat_test_real_non_d <- dat_test_real_male
    dat_test_find_discr <- dat_test_find_female
    am_model_x <- mod_am_f
    am_model_y <- mod_am_m
    sav_model_x <- mod_sav_f
    sav_model_y <- mod_sav_m
    risk_model_x <- mod_risk_f
    risk_model_y <- mod_risk_m
  }else{
    dat_test_warped_discr <- dat_test_real_male
    dat_test_real_discr <- dat_test_real_male
    dat_test_real_non_d <- dat_test_real_female
    dat_test_find_discr <- dat_test_find_male
    am_model_x <- mod_am_m
    am_model_y <- mod_am_f
    sav_model_x <- mod_sav_m
    sav_model_y <- mod_sav_f
    risk_model_x <- mod_risk_m
    risk_model_y <- mod_risk_f
  }
  
  # Warp female "Amount" values to male model
  dat_test_warped_discr[,"Amount"] <- warp_new_data(dat_new = dat_test_warped_discr, 
                                                    model_x = am_model_x, 
                                                    model_y = am_model_y,
                                                    target = "Amount")
  
  # Warp female "Saving" values to male model
  dat_test_warped_discr[,"Saving"] <- warp_new_data(dat_new = dat_test_warped_discr, 
                                                    model_x = sav_model_x, 
                                                    model_y = sav_model_y,
                                                    target = "Saving")
  
  # Warp female "Risk" values to male model
  dat_test_warped_discr[,"Risk"] <- warp_new_data(dat_new = dat_test_warped_discr, 
                                                  model_x = risk_model_x, 
                                                  model_y = risk_model_y,
                                                  target = "Risk")
  
  
  #----------------------------------#
  #### 7. fairadapt ####
  #----------------------------------#
  # Train fairadapt and predict test data, see german_credit_study.R
  # EDIT: Did this above in 2.a-II)
  # Now only formatting:
  
  dat_test_adapt_discr <- adapt_test
  dat_test_adapt_discr$Risk <- as.numeric(adapt_test$Risk)-1
  
  rownames(dat_test_adapt_discr) <- rownames(dat_test_warped_discr)
  # head(dat_test_warped_discr)
  # head(dat_test_adapt_discr)
  # head(dat_test_real_female)
  # head(dat_test_find_female)
  
  
  #----------------------------------#
  # 5. Pre-computations for results ####
  #----------------------------------#
  
  # Predict target for test data
  
  # Warped + adapt - Baseline
  if(!reverse_warping){
    pred_female_warped_baseline <- predict(mod_baseline, 
                                           newdata = dat_test_warped_discr, 
                                           type="response")
    pred_male_warped_baseline <- predict(mod_baseline, 
                                         newdata = dat_test_real_male, 
                                         type="response")
    
    pred_female_adapt_baseline <- predict(mod_baseline, 
                                          newdata = dat_test_adapt_discr, 
                                          type="response")
    pred_male_adapt_baseline <- predict(mod_baseline, 
                                        newdata = dat_test_real_male, 
                                        type="response")
  }else{
    pred_female_warped_baseline <- predict(mod_baseline, 
                                           newdata = dat_test_real_female, 
                                           type="response")
    pred_male_warped_baseline <- predict(mod_baseline, 
                                         newdata = dat_test_warped_discr, 
                                         type="response")
    
    pred_female_adapt_baseline <- predict(mod_baseline, 
                                          newdata = dat_test_real_female, 
                                          type="response")
    pred_male_adapt_baseline <- predict(mod_baseline, 
                                        newdata = dat_test_adapt_discr, 
                                        type="response")
  }
  
  # Warped + adapt
  if(!reverse_warping){
    pred_female_warped <- predict(mod_warped, 
                                  newdata = dat_test_warped_discr, 
                                  type="response")
    pred_male_warped <- predict(mod_warped, 
                                newdata = dat_test_real_male, 
                                type="response")
    
    pred_female_adapt <- predict(mod_adapt, 
                                 newdata = dat_test_adapt_discr, 
                                 type="response")
    pred_male_adapt <- predict(mod_adapt, 
                               newdata = dat_test_real_male, 
                               type="response")
  }else{
    pred_female_warped <- predict(mod_warped, 
                                  newdata = dat_test_real_female, 
                                  type="response")
    pred_male_warped <- predict(mod_warped, 
                                newdata = dat_test_warped_discr, 
                                type="response")
    
    pred_female_adapt <- predict(mod_adapt, 
                                 newdata = dat_test_real_female, 
                                 type="response")
    pred_male_adapt <- predict(mod_adapt, 
                               newdata = dat_test_adapt_discr, 
                               type="response")
  }
  
  # Real and FiND
  pred_female_real <- predict(mod_real, newdata = dat_test_real_female, type="response")
  pred_female_find <- predict(mod_find, newdata = dat_test_find_female, type="response")
  pred_male_real <- predict(mod_real, newdata = dat_test_real_male, type="response")
  pred_male_find <- predict(mod_find, newdata = dat_test_find_male, type="response")
  
  # TODO Use sex in real world model for comparison or not?
  
  #--------------#
  # 2) Mapping
  #   a) Features: Which features vary most between the 2 worlds?
  
  # MSEs of normalized values
  mse_vec <- mse_func_col(dat_test_warped_discr, dat_test_real_discr, cols=cols)
  mse_vec_adapt <- mse_func_col(dat_test_adapt_discr, dat_test_real_discr, cols=cols)
  
  
  # # Plots of distributions in the 2 worlds
  # plot_warped(dat_test_warped_discr, dat_test_real_discr, "Age")
  # plot_warped(dat_test_warped_discr, dat_test_real_discr, "Amount",ylim=c(0,0.0003))
  # plot_warped(dat_test_warped_discr, dat_test_real_discr, "Saving")
  # 
  # # Saving: Summary
  # summary(dat_test_warped_discr$Saving)
  # summary(dat_test_real_discr$Saving)
  
  #----------#
  #   b) Observations: Which observations change the most between the 2 worlds?
  
  dat_eval_map <- dat_test_real_discr[,cols]
  dat_eval_map$a_warped <- dat_test_warped_discr[,"Amount"]
  dat_eval_map$a_adapt <- dat_test_adapt_discr[,"Amount"]
  dat_eval_map$a_non_d <- dat_test_real_non_d[,"Amount"]
  dat_eval_map$s_warped <- dat_test_warped_discr[,"Saving"]
  dat_eval_map$s_adapt <- dat_test_adapt_discr[,"Saving"]
  dat_eval_map$s_non_d <- dat_test_real_non_d[,"Saving"]
  dat_eval_map$mse_row_wr <- mse_func_row(dat_test_warped_discr, dat_test_real_discr, cols=cols)
  dat_eval_map$mse_row_ar <- mse_func_row(dat_test_adapt_discr, dat_test_real_discr, cols=cols)
  dat_eval_map$mse_row_fr <- mse_func_row(dat_test_find_discr, dat_test_real_discr, cols=cols)
  dat_eval_map$mse_row_fw <- mse_func_row(dat_test_find_discr, dat_test_warped_discr, cols=cols)
  dat_eval_map$mse_row_fa <- mse_func_row(dat_test_find_discr, dat_test_adapt_discr, cols=cols)
  dat_eval_map[,-1] <- round(dat_eval_map[,-1], 3)
  # head(dat_eval_map[order(dat_eval_map$mse, decreasing=TRUE),])
  
  # mod_warp_change_sim <- gam(mse~s(Age) + s(Amount) + Saving, dat=dat_eval_map)
  # summary(mod_warp_change_sim)
  # plot(mod_warp_change_sim, pages=1)
  
  
  #------------#
  # 3) ML model
  #   b) Compare predictions in the 2 worlds => similar to 2b)
  dat_eval_pred <- dat_test_real_discr[,cols]
  dat_eval_pred$pf_real <- round(pred_female_real,4)
  dat_eval_pred$pf_warped <- round(pred_female_warped,4)
  dat_eval_pred$pf_adapt <- round(pred_female_adapt,4)
  dat_eval_pred$pf_find <- round(pred_female_find,4)
  dat_eval_pred$pm_real <- round(pred_male_real,4)
  dat_eval_pred$pm_warped <- round(pred_male_warped,4)
  dat_eval_pred$pm_adapt <- round(pred_male_adapt,4)
  dat_eval_pred$pm_find <- round(pred_male_find,4)
  dat_eval_pred$diff_mff <- round(pred_male_find-pred_female_find,4)
  dat_eval_pred$diff_mfw <- round(pred_male_warped-pred_female_warped,4)
  dat_eval_pred$diff_mfa <- round(pred_male_adapt-pred_female_adapt,4)
  dat_eval_pred$diff_mfr <- round(pred_male_real-pred_female_real,4)
  dat_eval_pred$diff_f_wr <- round(pred_female_warped-pred_female_real,4)
  dat_eval_pred$diff_f_ar <- round(pred_female_adapt-pred_female_real,4)
  dat_eval_pred$diff_m_wr <- round(pred_male_warped-pred_male_real,4)
  dat_eval_pred$diff_m_ar <- round(pred_male_adapt-pred_male_real,4)
  dat_eval_pred$diff_f_fr <- round(pred_female_find-pred_female_real,4)
  dat_eval_pred$diff_m_fr <- round(pred_male_find-pred_male_real,4)
  dat_eval_pred$diff_f_fw <- round(pred_female_find-pred_female_warped,4)
  dat_eval_pred$diff_f_fa <- round(pred_female_find-pred_female_adapt,4)
  dat_eval_pred$diff_m_fw <- round(pred_male_find-pred_male_warped,4)
  dat_eval_pred$diff_m_fa <- round(pred_male_find-pred_male_adapt,4)
  
  dat_eval_pred$diff_f_wr_base <- round(pred_female_warped_baseline-pred_female_real,4)
  dat_eval_pred$diff_f_ar_base <- round(pred_female_adapt_baseline-pred_female_real,4)
  dat_eval_pred$diff_m_wr_base <- round(pred_male_warped_baseline-pred_male_real,4)
  dat_eval_pred$diff_m_ar_base <- round(pred_male_adapt_baseline-pred_male_real,4)
  # head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=TRUE),])
  
  
  # # Compare prediction diffs between worlds - for each sex
  # boxplot(dat_eval_pred$diff_m_wr, dat_eval_pred$diff_f_wr,
  #         main="Diff in preds between worlds (warped-real)")
  # axis(1,at=c(1,2) ,labels=c("male", "female"), las=1)
  # 
  # t.test(dat_eval_pred$diff_m_wr)$p.value
  # t.test(dat_eval_pred$diff_f_wr)$p.value
  
  # => Preds for male do not change (significantly). Wouldn't have been bad if 
  #   that would have happened, though
  
  # # Compare prediction diffs between sex's - for each world
  # boxplot(dat_eval_pred$diff_mfr, dat_eval_pred$diff_mfw, main="Diff in preds between PAs (male-female)")
  # axis(1,at=c(1,2) ,labels=c("real world", "warped world"), las=1)
  # 
  # t.test(dat_eval_pred$diff_mfr)$p.value
  # t.test(dat_eval_pred$diff_mfw)$p.value
  
  # # => Yes, no discrimination in warped world!!!!! (no relevant discrimination)
  # #   (for age not sex-dependent)
  # 
  # t.test(dat_eval_pred$pf_warped - dat_eval_pred$pm_real)$p.value
  
  # => Together with not changing male preds: Warped females = real males :-)
  
  out_list <- list()
  
  out_list$fair_check_m <- fair_check_m
  out_list$fair_score_m <- fair_score_m
  out_list$dat_real <- dat_real
  out_list$dat_find <- dat_find
  out_list$dat_warped <- dat_warped
  out_list$dat_adapt <- dat_adapt
  # list_real[[m]] <- dat_real
  # list_find[[m]] <- dat_find
  # list_warped[[m]] <- dat_warped
  # list_adapt[[m]] <- dat_adapt
  
  out_list$mse_vec <- mse_vec
  out_list$mse_vec_adapt <- mse_vec_adapt
  out_list$dat_eval_map <- dat_eval_map
  out_list$dat_eval_pred <- dat_eval_pred
  # list_mse[[m]] <- mse_vec
  # list_mse_adapt[[m]] <- mse_vec_adapt
  # list_map[[m]] <- dat_eval_map
  # list_pred[[m]] <- dat_eval_pred
  
  out_list$dat_test_real_male <- dat_test_real_male
  out_list$dat_test_real_female <- dat_test_real_female
  # list_test_real_male[[m]] <- dat_test_real_male
  # list_test_real_female[[m]] <- dat_test_real_female
  
  if(!reverse_warping){
    out_list$dat_test_find_female <- dat_test_find_female
    out_list$dat_test_warped_discr <- dat_test_warped_discr
    out_list$dat_test_adapt_discr <- dat_test_adapt_discr
    # list_test_find_female[[m]] <- dat_test_find_female
    # list_test_warped_female[[m]] <- dat_test_warped_discr
    # list_test_adapt_female[[m]] <- dat_test_adapt_discr
  }else{
    out_list$dat_test_find_male <- dat_test_find_male
    out_list$dat_test_warped_discr <- dat_test_warped_discr
    out_list$dat_test_adapt_discr <- dat_test_adapt_discr
    # list_test_find_male[[m]] <- dat_test_find_male
    # list_test_warped_male[[m]] <- dat_test_warped_discr
    # list_test_adapt_male[[m]] <- dat_test_adapt_discr
  }
  return(out_list)
}


result <- mclapply(seq_len(M_iter),
                   myfun,
                   mc.cores=num_cores)
# str(result, max.level=1)
# str(result[[1]], max.level=1)

save(result, file="results/sim/temp.RData")
#load("results/sim/temp.RData")

for(m in seq_len(M_iter)){
  cat("M =", m, "\n")
  #tryCatch({
    # if(!age_sex){
      fair_check_mat[m,] <- result[[m]]$fair_check_m
      fair_score_mat[m,] <- result[[m]]$fair_score_m
    #}
    list_real[[m]] <- result[[m]]$dat_real
    list_find[[m]] <- result[[m]]$dat_find
    list_warped[[m]] <- result[[m]]$dat_warped
    list_adapt[[m]] <- result[[m]]$dat_adapt
    list_mse[[m]] <- result[[m]]$mse_vec
    list_mse_adapt[[m]] <- result[[m]]$mse_vec_adapt
    list_map[[m]] <- result[[m]]$dat_eval_map
    list_pred[[m]] <- result[[m]]$dat_eval_pred
    list_test_real_male[[m]] <- result[[m]]$dat_test_real_male
    list_test_real_female[[m]] <- result[[m]]$dat_test_real_female
    if(!reverse_warping){
      list_test_find_female[[m]] <- result[[m]]$dat_test_find_female
      list_test_warped_female[[m]] <- result[[m]]$dat_test_warped_discr
      list_test_adapt_female[[m]] <- result[[m]]$dat_test_adapt_discr
    }else{
      list_test_find_male[[m]] <- result[[m]]$dat_test_find_male
      list_test_warped_male[[m]] <- result[[m]]$dat_test_warped_discr
      list_test_adapt_male[[m]] <- result[[m]]$dat_test_adapt_discr
    }
  # }, error = function(e) {
  #   print("An error occurred:")
  #   print(e)
  # })
}

#----------------------------------#
# 6. Save data for later use in results_sim.R ####
#----------------------------------#

file_name <- paste0("results/sim/simulation_", date_start,
                    "_rev_warp_", reverse_warping,
                    "_age_sex_",age_sex,
                    "_M_",M_iter,".RData")

if(!reverse_warping){
  save(#list_real, list_find, list_warped, # we don't use those at the moment..
    list_mse, list_mse_adapt, list_map, list_pred,
    list_test_real_male, list_test_real_female, # perhaps only m=1 here
    list_test_find_female, list_test_warped_female, list_test_adapt_female,
    fair_check_mat, fair_score_mat,
    M_iter,
    config,
    file=file_name)
}else{
  save(#list_real, list_find, list_warped, # we don't use those at the moment..
    list_mse, list_mse_adapt, list_map, list_pred,
    list_test_real_male, list_test_real_female, # perhaps only m=1 here
    list_test_find_male, list_test_warped_male, list_test_adapt_male,
    fair_check_mat, fair_score_mat,
    M_iter,
    config,
    file=file_name)
}

cat("saved as", file_name, "\n")

print(Sys.time() - start_time)

cat(rep("=", 50), "\n", sep = "")
cat("End simulation with ",
    # M_iter, " iterations, age_sex = ", age_sex,
    # ", reverse warping = ", reverse_warping, ", n_train = ",
    # n_train, ", n_test = ", n_test, ", epsilon = ", epsilon,
    " \n", sep = "")
print(config)
cat(rep("=", 50), "\n", sep = "")

# TODO: Add adapt to sim_results as well