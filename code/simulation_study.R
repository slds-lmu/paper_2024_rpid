library(mgcv)
library(simcausal)
set.seed(940984)
source("code/func_help.R")

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


M_iter <- 1000 # number iterations of simulation study
n_train <- 10000 # size of training set
n_test <- 1000 # size of test set
age_sex <- TRUE # TRUE: Age is simulated as sex-dependent
reverse_warping <- FALSE # TRUE: Warp male to female population
rndseed <- 1425 
rndseed_test <- 1427
cols <- c("Age", "Amount", "Saving")


cat(rep("=",50), "\n", sep="")
cat("Start simulation with ", M_iter, " iterations, age_sex = ", age_sex,
    ", reverse warping = ", reverse_warping, ", n_train = ", n_train, ", n_test = ", n_test, " \n", sep="")
cat(rep("=",50), "\n", sep="")


list_real <- list()
list_find <- list()
list_warped <- list()
list_mse <- list()
list_map <- list()
list_pred <- list()
list_test_real_male <- list()
list_test_real_female <- list()
if(!reverse_warping){
  list_test_find_female <- list()
  list_test_warped_female <- list()
}else{
  list_test_find_male <- list()
  list_test_warped_male <- list()
}

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

# Stupid intervention, which does not do anything, just 
#   for getting "real world" data in the simulation
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

for(m in seq_len(M_iter)){
  
  cat("M =", m, "\n")
  
  # Simulate data
  dat_find_act <- simcausal::sim(DAG = Mfind_act, actions = c("real_world","find_world"),
                                 n = n_train, rndseed = rndseed+m, verbose = FALSE)
  
  # Separate both worlds in one data.frame, respectively
  dat_find <- dat_find_act$find_world[,-1]
  dat_find$Age <- round(dat_find$Age)
  dat_find$Sex <- factor(dat_find$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_find$Risk <- factor(dat_find$Risk, levels = c(0:1), labels = c("bad", "good"))
  
  dat_real <- dat_find_act$real_world[,-1]
  dat_real$Age <- round(dat_real$Age)
  dat_real$Sex <- factor(dat_real$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_real$Risk <- factor(dat_real$Risk, levels = c(0:1), labels = c("bad", "good"))
  
  # gender-rows
  male_rows <- dat_real$Sex == "male"
  female_rows <- dat_real$Sex != "male"
  
  # sub-data sets for each gender
  dat_real_m <- dat_real[male_rows,]
  dat_real_f <- dat_real[female_rows,]

  #----------------------------------#
  #### 2. Warp real world data to warped world ####
  #----------------------------------#
  
  ### Amount
  # Male and female model for warping
  mod_am_f <- glm(Amount ~ 1 + Age, family = Gamma(link="log"), data = dat_real_f)
  mod_am_m <- glm(Amount ~ 1 + Age, family = Gamma(link="log"), data = dat_real_m)

  
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

  
  list_real[[m]] <- dat_real
  list_find[[m]] <- dat_find
  list_warped[[m]] <- dat_warped
  
  
  #----------------------------------#
  # Sanity check if sim is as expected
  #----------------------------------#
  
  #----------------------------------#
  # Models in different worlds  
  #----------------------------------#
  
  
  mod_real <- glm(Risk ~ ., family = binomial, data = dat_real)
  mod_warped <- glm(Risk ~ . - Sex, family = binomial, data = dat_warped)
  mod_warped_sex <- glm(Risk ~ ., family = binomial, data = dat_warped)
  mod_find <- glm(Risk ~ . - Sex, family = binomial, data = dat_find)
  
  if(!reverse_warping){
    # baseline: instead of using warped data, just use male data
    mod_baseline <- glm(Risk ~ Age+Amount+Saving, family = binomial, data = dat_real_m)
  }else{
    # baseline: instead of using warped data, just use female data
    mod_baseline <- glm(Risk ~ Age+Amount+Saving, family = binomial, data = dat_real_f)
  }  
  
  summary(mod_real)$coef
  summary(mod_warped_sex)$coef
  summary(mod_find)$coef
  # 
  
  
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
  
  dat_test_real_male$Sex <- factor(dat_test_real_male$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_test_real_female$Sex <- factor(dat_test_real_female$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_test_find_male$Sex <- factor(dat_test_find_male$Sex, levels = c(0:1), labels = c("female", "male"))
  dat_test_find_female$Sex <- factor(dat_test_find_female$Sex, levels = c(0:1), labels = c("female", "male"))
  
  dat_test_real_male$Age <- round(dat_test_real_male$Age)
  dat_test_real_female$Age <- round(dat_test_real_female$Age)
  dat_test_find_male$Age <- round(dat_test_find_male$Age)
  dat_test_find_female$Age <- round(dat_test_find_female$Age)
  
  #----------------------------------#
  # 4. Warp test data with training data warping model ####
  #----------------------------------#
  
  if(!reverse_warping){
    # "discr" means the "discriminated PA", i.e., the one that differs between 
    #   real and find world
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
  # 5. Pre-computations for results ####
  #----------------------------------#

  # Predict target for test data
  
  # Warped - Baseline
  if(!reverse_warping){
    pred_female_warped_baseline <- predict(mod_baseline, newdata = dat_test_real_female, type="response")
    pred_male_warped_baseline <- predict(mod_baseline, newdata = dat_test_real_male, type="response")
  }else{
    pred_female_warped_baseline <- predict(mod_baseline, newdata = dat_test_real_female, type="response")
    pred_male_warped_baseline <- predict(mod_baseline, newdata = dat_test_real_male, type="response")
  }
  
  # Warped
  if(!reverse_warping){
    pred_female_warped <- predict(mod_warped, newdata = dat_test_warped_discr, type="response")
    pred_male_warped <- predict(mod_warped, newdata = dat_test_real_male, type="response")
  }else{
    pred_female_warped <- predict(mod_warped, newdata = dat_test_real_female, type="response")
    pred_male_warped <- predict(mod_warped, newdata = dat_test_warped_discr, type="response")
  }
  
  # Real and FiND
  pred_female_real <- predict(mod_real, newdata = dat_test_real_female, type="response")
  pred_female_find <- predict(mod_find, newdata = dat_test_find_female, type="response")
  pred_male_real <- predict(mod_real, newdata = dat_test_real_male, type="response")
  pred_male_find <- predict(mod_find, newdata = dat_test_find_male, type="response")
  

  #--------------#
  # 2) Mapping
  #   a) Features: Which features vary most between the 2 worlds?
  
  # MSEs of normalized values
  mse_vec <- mse_func_col(dat_test_warped_discr, dat_test_real_discr, cols=cols)
  

  #----------#
  #   b) Observations: Which observations change the most between the 2 worlds?
  
  dat_eval_map <- dat_test_real_discr[,cols]
  dat_eval_map$a_warped <- dat_test_warped_discr[,"Amount"]
  dat_eval_map$a_non_d <- dat_test_real_non_d[,"Amount"]
  dat_eval_map$s_warped <- dat_test_warped_discr[,"Saving"]
  dat_eval_map$s_non_d <- dat_test_real_non_d[,"Saving"]
  dat_eval_map$mse_row_wr <- mse_func_row(dat_test_warped_discr, dat_test_real_discr, cols=cols)
  dat_eval_map$mse_row_fr <- mse_func_row(dat_test_find_discr, dat_test_real_discr, cols=cols)
  dat_eval_map$mse_row_fw <- mse_func_row(dat_test_find_discr, dat_test_warped_discr, cols=cols)
  dat_eval_map[,-1] <- round(dat_eval_map[,-1], 3)

  #------------#
  # 3) ML model
  #   b) Compare predictions in the 2 worlds => similar to 2b)
  dat_eval_pred <- dat_test_real_discr[,cols]
  dat_eval_pred$pf_real <- round(pred_female_real,4)
  dat_eval_pred$pf_warped <- round(pred_female_warped,4)
  dat_eval_pred$pf_find <- round(pred_female_find,4)
  dat_eval_pred$pm_real <- round(pred_male_real,4)
  dat_eval_pred$pm_warped <- round(pred_male_warped,4)
  dat_eval_pred$pm_find <- round(pred_male_find,4)
  dat_eval_pred$diff_mff <- round(pred_male_find-pred_female_find,4)
  dat_eval_pred$diff_mfw <- round(pred_male_warped-pred_female_warped,4)
  dat_eval_pred$diff_mfr <- round(pred_male_real-pred_female_real,4)
  dat_eval_pred$diff_f_wr <- round(pred_female_warped-pred_female_real,4)
  dat_eval_pred$diff_m_wr <- round(pred_male_warped-pred_male_real,4)
  dat_eval_pred$diff_f_fr <- round(pred_female_find-pred_female_real,4)
  dat_eval_pred$diff_m_fr <- round(pred_male_find-pred_male_real,4)
  dat_eval_pred$diff_f_fw <- round(pred_female_find-pred_female_warped,4)
  dat_eval_pred$diff_m_fw <- round(pred_male_find-pred_male_warped,4)
  
  dat_eval_pred$diff_f_wr_base <- round(pred_female_warped_baseline-pred_female_real,4)
  dat_eval_pred$diff_m_wr_base <- round(pred_male_warped_baseline-pred_male_real,4)

  list_mse[[m]] <- mse_vec  
  list_map[[m]] <- dat_eval_map
  list_pred[[m]] <- dat_eval_pred

  list_test_real_male[[m]] <- dat_test_real_male
  list_test_real_female[[m]] <- dat_test_real_female
  if(!reverse_warping){
    list_test_find_female[[m]] <- dat_test_find_female
    list_test_warped_female[[m]] <- dat_test_warped_discr
  }else{
    list_test_find_male[[m]] <- dat_test_find_male
    list_test_warped_male[[m]] <- dat_test_warped_discr
    
  }
   
}

#----------------------------------#
# 6. Save data for later use in results_sim.R ####
#----------------------------------#

file_name <- paste0("results/sim/simulation_",Sys.Date(),

                    "_rev_warp_", reverse_warping,
                    "_age_sex_",age_sex,
                    "_M_",M_iter,".RData")

if(!reverse_warping){
  save(#list_real, list_find, list_warped, # we don't use those at the moment..
    list_mse, list_map, list_pred,
    list_test_real_male, list_test_real_female, 
    list_test_find_female, list_test_warped_female,
    M_iter,
    file=file_name)
}else{
  save(#list_real, list_find, list_warped, # we don't use those at the moment..
    list_mse, list_map, list_pred,
    list_test_real_male, list_test_real_female, 
    list_test_find_male, list_test_warped_male,
    M_iter,
    file=file_name)
}

cat("saved as", file_name, "\n")

print(Sys.time() - start_time)
