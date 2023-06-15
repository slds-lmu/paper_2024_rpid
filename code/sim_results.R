library(mgcv)
source("code/func_help.R")

#----------------------------------#
#### Structure of this code ########
#----------------------------------#
# 0. Load data
# 1. Produce numbers and plots for simulation study
#----------------------------------#



#----------------------------------#
#### 0. Load data ####
#----------------------------------#
save_plots <- FALSE
rq_age_sex <- FALSE # sex-dependent age
reverse_warping <- FALSE # reverse warping

# NOTE: You have to run the simulation study in simulation_study.R first

file_name <- "results/sim/simulation_2023-06-12_rev_warp_FALSE_age_sex_FALSE_M_1000.RData"
file_rev <- "results/sim/simulation_2023-06-12_rev_warp_TRUE_age_sex_FALSE_M_1000.RData" 
file_rq_age_sex <- "results/sim/simulation_2023-06-13_rev_warp_FALSE_age_sex_TRUE_M_1000.RData"
if(rq_age_sex){
  load(file_rq_age_sex)
  cat("loading data for rq_age_sex \n")
}else if(reverse_warping){
  load(file_rev)
  cat("loading data for reverse warping \n")
}else{
  load(file_name)
  cat("loading data for RQ1 \n")
}

#----------------------------------#
#### 1. Produce numbers and plots for simulation study ####
#----------------------------------#

# 1) Plot Amount over all iterations
# m <- 1
dat_test_real_non_d <- dat_test_real_discr <- dat_test_find_discr <- dat_test_warped_discr <- vector(length=0)
for(m in seq_len(M_iter)){
  
  if(reverse_warping){
    dat_test_real_non_d <- c(dat_test_real_non_d, list_test_real_female[[m]]$Amount)
    dat_test_real_discr <- c(dat_test_real_discr, list_test_real_male[[m]]$Amount)
    dat_test_find_discr <- c(dat_test_find_discr, list_test_find_male[[m]]$Amount)
    dat_test_warped_discr <- c(dat_test_warped_discr, list_test_warped_male[[m]]$Amount)
  }else{
    dat_test_real_non_d <- c(dat_test_real_non_d, list_test_real_male[[m]]$Amount)
    dat_test_real_discr <- c(dat_test_real_discr, list_test_real_female[[m]]$Amount)
    dat_test_find_discr <- c(dat_test_find_discr, list_test_find_female[[m]]$Amount)
    dat_test_warped_discr <- c(dat_test_warped_discr, list_test_warped_female[[m]]$Amount)
  }
}

if(save_plots){
  if(rq_age_sex){
    pdf("plots/rq_age_sex-1-1.pdf")
  }else if(reverse_warping){
    pdf("plots/rq_rev_warp-1-1.pdf")
  }else{
    pdf("plots/rq1-1.pdf")
  }
}
plot_dens(x=list(dat_test_real_discr, 
                 dat_test_warped_discr,
                 dat_test_find_discr,
                 dat_test_real_non_d),
          leg = c("Real Female", "Warped Female", "Find Female", "Real Male"), 
          xlab = "Amount",
          main = "Real, warped, and FiND world", legend_position = "topright",
          ylim=c(0,0.0003))
if(save_plots){
  dev.off()
}


btr <- bts <- ksa <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  if(reverse_warping){
    dat_test_real_non_d <- list_test_real_female[[m]]
    dat_test_real_discr <- list_test_real_male[[m]]
    dat_test_find_discr <- list_test_find_male[[m]]
    dat_test_warped_discr <- list_test_warped_male[[m]]
  }else{
    dat_test_real_non_d <- list_test_real_male[[m]]
    dat_test_real_discr <- list_test_real_female[[m]]
    dat_test_find_discr <- list_test_find_female[[m]]
    dat_test_warped_discr <- list_test_warped_female[[m]]
  }
  
  dat_test_real <- rbind(dat_test_real_discr, dat_test_real_non_d)
  dat_test_warped <- rbind(dat_test_warped_discr, dat_test_real_non_d)
  dat_test_find <- rbind(dat_test_find_discr, dat_test_real_non_d)
  
  # Amount
  ksa[m] <- ks.test(dat_test_warped_discr$Amount, dat_test_find_discr$Amount)$p.value
  
  # Saving
  if(!reverse_warping){
    sav_thresholded <- func_thresh(dat_test_warped, "Saving")
  }else{
    sav_thresholded <- func_thresh(dat_test_warped, "Saving", from="male", to="female")
  }

  bts[m] <- binom.test(table(sav_thresholded)[2:1], p = mean(dat_test_find_discr$Saving))$p.value
  

  # Risk
  if(!reverse_warping){
    risk_thresholded <- func_thresh(dat_test_warped, "Risk")
  }else{
    risk_thresholded <- func_thresh(dat_test_warped, "Risk", from="male", to="female")
  }

  
  btr[m] <- binom.test(table(risk_thresholded)[2:1], p = mean(dat_test_find_discr$Risk))$p.value
 
}

# Plot 2
if(save_plots){
  if(rq_age_sex){
    pdf("plots/rq_age_sex-1-2.pdf")
  }else if(reverse_warping){
    pdf("plots/rq_rev_warp-1-2.pdf")
  }else{
    pdf("plots/rq1-2.pdf")
  }
}
boxplot(ksa, bts, btr, xlab=NULL, 
        main = "Test on warped == FiND world distributions", ylab="p-value")
axis(1,at=c(1:3) ,labels=c("amount", "saving","risk"), las=1)
abline(h=0.05)
if(save_plots){
  dev.off()
}



# Numbers for RQ1
mean(ksa<0.05)
mean(bts<0.05)
mean(btr<0.05)


#----------------------------------#
#### Find most discriminated individuals #
#----------------------------------#

# "Can individuals be correctly identified that profit or suffer most from 
# unfair discrimination in the real world?"

med_rank <- cor_ranks <- vector(length = M_iter)
range_rank <- matrix(0,ncol=2, nrow=M_iter)
md <- 50
for(m in seq_len(M_iter)){
  #m <- 1
  dat_eval_map <- list_map[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  
  #   Individuals that profit or suffer most from discrimination: 
  #   Largest gap between real and find world:
  if(reverse_warping){
    most_discr <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_fr, decreasing=TRUE),]))
  }else{
    most_discr <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_fr, decreasing=TRUE),]))
  }
  
  #   "can be identified..": as most discriminated can be seen those where real
  #     and warped world differs most:
  if(reverse_warping){
    most_discr_warped <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_wr, decreasing=TRUE),]))
  }else{
    most_discr_warped <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=TRUE),]))
  }
  
  # data frame with IDs and both ranks
  discr_df_find <- data.frame(ID = most_discr, rank_find = seq_len(length(most_discr)))
  
  discr_df_warped <- data.frame(ID = most_discr_warped, rank_warped = seq_len(length(most_discr)))
  discr_df <- merge(discr_df_find, discr_df_warped, by = "ID")
  

  # Comparison of both vectors:
  cor_ranks[m] <- cor(discr_df$rank_find, discr_df$rank_warped)

  top10 <- most_discr[1:md]
  med_rank[m] <- median(discr_df$rank_warped[top10])
  range_rank[m,] <- range(discr_df$rank_warped[top10])
}

boxplot(med_rank, range_rank[,1], range_rank[,2])
quantile(med_rank)
mean(med_rank<md*2)



# Plot (2)
if(save_plots){
  if(rq_age_sex){
    pdf("plots/rq_age_sex-2-2.pdf")
  }else if(reverse_warping){
    pdf("plots/rq_rev_warp-2-2.pdf")
  }else{
    pdf("plots/rq2-2.pdf")
  }
}
boxplot(cor_ranks, main = "Correlation of ranks of pred diffs", ylab="corr")
if(save_plots){
  dev.off()
}


t_wr_est <- cor_wr <- t_wr  <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  #m <- 1
  dat_eval_pred <- list_pred[[m]]
  
  wr <- dat_eval_pred$diff_f_wr
  fr <- dat_eval_pred$diff_f_fr

  t_wr[m] <- t.test(wr, fr)$p.value
  t_wr_est[m] <- diff(t.test(wr, fr)$estimate)


}

mean(t_wr_est)
t.test(t_wr_est) #
mean(t_wr)
1-mean(t_wr<0.05)
mean(t_wr_est[t_wr<0.05]) 
mean(t_wr_est)


md <- 10
top10 <- most_discr[1:md]
top10w <- most_discr_warped[1:md]

bottom10 <- most_discr[(length(most_discr)-md):length(most_discr)]
bottom10w <- most_discr_warped[(length(most_discr)-md):length(most_discr)]



# Plot (1)
if(save_plots){
  if(rq_age_sex){
    pdf("plots/rq_age_sex-2-1.pdf")
  }else if(reverse_warping){
    pdf("plots/rq_rev_warp-2-1.pdf")
  }else{
    pdf("plots/rq2-1.pdf")
  }
}
if(reverse_warping){
  boxplot(dat_eval_pred$diff_m_wr, dat_eval_pred$diff_m_fr, xlab=NULL, main="risk pred diff")
  axis(1,at=c(1:2) ,labels=c("male warped-real", "male find-real"), las=1)
  # Add hairlines connecting the corresponding values
  for (i in c(top10)){#, bottom10w)) {
    segments(1, dat_eval_pred$diff_m_wr[i], 2, dat_eval_pred$diff_m_fr[i], col = "gray", lty = "solid")
  }
}else{
  boxplot(dat_eval_pred$diff_f_wr, dat_eval_pred$diff_f_fr, xlab=NULL, main="Risk prediction differences",
          ylab="Difference")
  axis(1,at=c(1:2) ,labels=c("Female Warped-Real", "Female FiND-Real"), las=1)
  # Add hairlines connecting the corresponding values
  for (i in c(top10)){#, bottom10w)) {
    segments(1, dat_eval_pred$diff_f_wr[i], 2, dat_eval_pred$diff_f_fr[i], col = "gray", lty = "solid")
  }
}
if(save_plots){
  dev.off()
}
if(reverse_warping){
  boxplot(dat_eval_pred$diff_m_wr, dat_eval_pred$diff_m_fr, xlab=NULL, main="risk pred diff")
  axis(1,at=c(1:2) ,labels=c("female warped-real", "female find-real"), las=1)
  # Add hairlines connecting the corresponding values
  for (i in c(top10w)){#, bottom10)) {
    segments(1, dat_eval_pred$diff_m_wr[i], 2, dat_eval_pred$diff_m_fr[i], col = "gray", lty = "solid")
  }
}else{
  boxplot(dat_eval_pred$diff_f_wr, dat_eval_pred$diff_f_fr, xlab=NULL, main="risk pred diff")
  axis(1,at=c(1:2) ,labels=c("female warped-real", "female find-real"), las=1)
  # Add hairlines connecting the corresponding values
  for (i in c(top10w)){#, bottom10)) {
    segments(1, dat_eval_pred$diff_f_wr[i], 2, dat_eval_pred$diff_f_fr[i], col = "gray", lty = "solid")
  }
}


#----------------

t_real_est <- t_warped_est <- t_find_est <- t_real <- t_warped <- t_find <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  #m <- 1
  dat_eval_map <- list_map[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  t_real[m] <- t.test(dat_eval_pred$diff_mfr)$p.value
  t_real_est[m] <- t.test(dat_eval_pred$diff_mfr)$estimate
  t_warped[m] <- t.test(dat_eval_pred$diff_mfw)$p.value
  t_warped_est[m] <- t.test(dat_eval_pred$diff_mfw)$estimate
  t_find[m] <- t.test(dat_eval_pred$diff_mff)$p.value
  t_find_est[m] <- t.test(dat_eval_pred$diff_mff)$estimate
  
  # => No relevant discrimination in warped world 
  #   (and of course also not in find world, by design)
}

if(save_plots){
  if(rq_age_sex){
    pdf("plots/rq_age_sex-discr.pdf")
  }else if(reverse_warping){
    pdf("plots/rq_rev_warp-discr.pdf")
  }else{
    pdf("plots/rq-discr.pdf")
  }
}
# Compare prediction diffs between sex's - for each world
boxplot(dat_eval_pred$diff_mfr, dat_eval_pred$diff_mfw, dat_eval_pred$diff_mff,
        main="Diff in preds between PAs (male-female)", ylab="Pred diff")
axis(1,at=c(1:3) ,labels=c("real world", "warped world", "find world"), las=1)
if(save_plots){
  dev.off()
}

if(save_plots){
  if(rq_age_sex){
    pdf("plots/rq_age_sex-discr-p.pdf")
  }else if(reverse_warping){
    pdf("plots/rq_rev_warp-discr-p.pdf")
  }else{
    pdf("plots/rq-discr-p.pdf")
  }
}
boxplot(t_real, t_warped, t_find, main="p-values of diff between male and female risk preds",
        ylab="p-value")
axis(1,at=c(1:3) ,labels=c("real world", "warped world", "find world"), las=1)
if(save_plots){
  dev.off()
}

mean(t_warped)
mean(t_warped<0.05)
mean(t_real<0.05)
mean(t_find<0.05)

mean(t_real_est)
mean(t_warped_est)
mean(t_find_est)
t.test(t_real_est)
t.test(t_warped_est)

#######
# Reverse warping
#######


load(file_rev)
list_pred_rev <- list_pred
list_map_rev <- list_map
load(file_name)

ttf <- ttm <- ksd <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  dat_eval_pred_rev <- list_pred_rev[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  diff_warped_rev <- dat_eval_pred_rev$diff_mfw
  diff_warped <- dat_eval_pred$diff_mfw
  
  
  ksd[m] <- ks.test(diff_warped, diff_warped_rev)$p.value
}


#########
# Risk ranks
#########


cor_ranks <- vector(length = M_iter)
pf_warped_all <- pf_warped_all_rev <-pm_warped_all <- pm_warped_all_rev <- vector(length=0)
for(m in seq_len(M_iter)){
  
  dat_eval_map_rev <- list_map_rev[[m]]
  dat_eval_pred_rev <- list_pred_rev[[m]]
  
  dat_eval_map <- list_map[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  pf_warped_id <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_warped, decreasing=TRUE),]))
  pf_warped_id_rev <- as.numeric(rownames(dat_eval_pred_rev[order(dat_eval_pred_rev$pf_warped, decreasing=TRUE),]))
  
  # data frame with IDs and both ranks
  discr_df_find <- data.frame(ID = pf_warped_id, rank = seq_len(length(pf_warped_id)))
  
  discr_df_warped <- data.frame(ID = pf_warped_id_rev, rank_rev = seq_len(length(pf_warped_id_rev)))
  discr_df <- merge(discr_df_find, discr_df_warped, by = "ID")
  
  # Comparison of both vectors:
  cor_ranks[m] <- cor(discr_df$rank, discr_df$rank_rev)

  # Individual predictions
  pf_warped <- dat_eval_pred$pf_warped
  pf_warped_rev <- dat_eval_pred_rev$pf_warped
  ttf[m] <- t.test(pf_warped,pf_warped_rev)$p.value
  
  pm_warped <- dat_eval_pred$pm_warped
  pm_warped_rev <- dat_eval_pred_rev$pm_warped
  ttm[m] <- t.test(pm_warped,pm_warped_rev)$p.value
  
  pf_warped_all <- c(pf_warped_all, pf_warped)
  pf_warped_all_rev <- c(pf_warped_all_rev, pf_warped_rev)
  
  pm_warped_all <- c(pm_warped_all, pm_warped)
  pm_warped_all_rev <- c(pm_warped_all_rev, pm_warped_rev)
}

boxplot(cor_ranks, main = "Correlation of ranks of pred", ylab="corr")
mean(cor_ranks)
# => Ranks are equal in warped world, no matter if reverse warping or not..

boxplot(pf_warped, pf_warped_rev)
boxplot(pm_warped, pm_warped_rev)
boxplot(pf_warped_all, pf_warped_all_rev)
par(mfrow=c(2,1))
plot_dens(x = list(pm_warped_all, pm_warped_all_rev),
          leg = c("male warped", "male warped rev"), xlab = "risk prob", legend_position = "topleft",
          main = "Density of risk probs by different warping directions", single_panel = FALSE)
plot_dens(x = list(pf_warped_all, pf_warped_all_rev),
          leg = c("female warped", "female warped rev"), xlab = "risk prob", legend_position = "topleft",
          main = "Density of risk probs by different warping directions", single_panel = FALSE)

if(save_plots){
  pdf("plots/rq_rev_warp-general-level-shift.pdf")
}
plot_dens(x = list(pf_warped_all, pf_warped_all_rev, pm_warped_all, pm_warped_all_rev),
          leg = c("Female Warped f->m", "Female Warped m->f","Male Warped f->m", "Male Warped m->f"), 
          xlab = "Risk prediction", legend_position = "topleft",
          main = "Density of risk probs by different warping directions")
if(save_plots){
  dev.off()
}
# .. but the general level is not equal! Warping works in both cases, i.e.,
#   risk predictions are equal in both groups AND individual ranks are equal,
#   only the level changes


mean(ksd<0.05)
mean(ttf<0.05)
mean(ttm<0.05)
