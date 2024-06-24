library(mgcv)
library(xtable)
# setwd("C:/Users/ua341au/Dropbox/Documents/GitHub/paper_2023_causal_fairness")
source("code/func_help.R")

#----------------------------------#
#### Structure of this code ########
#----------------------------------#
# 0. Load data
# 1. RQ1 etc.
#----------------------------------#



#----------------------------------#
#### 0. Load data ####
#----------------------------------#
save_plots <- FALSE
height <- 5
rq4 <- FALSE # sex-dependent age
reverse_warping <- TRUE # reverse warping

#file_name <- "results/sim/simulation_2023-06-12_rev_warp_FALSE_age_sex_FALSE_M_1000.RData"
file_name <- "results/sim/simulation_2024-03-19_rev_warp_FALSE_age_sex_FALSE_M_1000.RData"
#file_rev <- "results/sim/simulation_2023-06-12_rev_warp_TRUE_age_sex_FALSE_M_1000.RData" #"results/sim/simulation_2023-06-08_rev_warp_TRUE_age_sex_FALSE_M_1000.RData"
file_rev <- "results/sim/simulation_2024-03-19_rev_warp_TRUE_age_sex_FALSE_M_1000.RData"
#file_rq4 <- "results/sim/simulation_2023-06-13_rev_warp_FALSE_age_sex_TRUE_M_1000.RData"
file_rq4 <- "results/sim/simulation_2024-03-20_rev_warp_FALSE_age_sex_TRUE_M_1000.RData"
# file_test <- "results/sim/simulation_2023-08-31_rev_warp_FALSE_age_sex_FALSE_M_1000.RData"
# load(file_test)
if(rq4){
  load(file_rq4)
  # load("results/sim/simulation_2023-06-06_age_sex_TRUE_M_1000.RData") # smaller train and test set
  cat("loading data for RQ4 \n")
}else if(reverse_warping){
  load(file_rev)
  cat("loading data for reverse warping \n")
}else{
  # load("results/sim/simulation_M1000_230601.RData")
  load(file_name)
  # load("results/sim/simulation_2023-06-06_age_sex_FALSE_M_1000.RData")
  cat("loading data for RQ1-3 \n")
}

#----------------------------------#
#### 1. RQ1 - good approximation of FiND world via warping? ####
#----------------------------------#

# 1) Plot Amount over all iterations
# m <- 1
dat_test_real_non_d <- dat_test_real_discr <- dat_test_find_discr <- dat_test_warped_discr <- dat_test_adapt_discr <- vector(length=0)
for(m in seq_len(M_iter)){
  
  if(reverse_warping){
    dat_test_real_non_d <- c(dat_test_real_non_d, list_test_real_female[[m]]$Amount)
    dat_test_real_discr <- c(dat_test_real_discr, list_test_real_male[[m]]$Amount)
    dat_test_find_discr <- c(dat_test_find_discr, list_test_find_male[[m]]$Amount)
    dat_test_warped_discr <- c(dat_test_warped_discr, list_test_warped_male[[m]]$Amount)
    dat_test_adapt_discr <- c(dat_test_adapt_discr, list_test_adapt_male[[m]]$Amount)
  }else{
    dat_test_real_non_d <- c(dat_test_real_non_d, list_test_real_male[[m]]$Amount)
    dat_test_real_discr <- c(dat_test_real_discr, list_test_real_female[[m]]$Amount)
    dat_test_find_discr <- c(dat_test_find_discr, list_test_find_female[[m]]$Amount)
    dat_test_warped_discr <- c(dat_test_warped_discr, list_test_warped_female[[m]]$Amount)
    dat_test_adapt_discr <- c(dat_test_adapt_discr, list_test_adapt_female[[m]]$Amount)
  }
}
# plot_warped(dat_test_warped_discr, dat_test_real_discr, "Amount",ylim=c(0,0.0003))

leg <- c("Real Female", "Warped Female", "Adapt Female", "Find Female", "Real Male")
if(save_plots){
  if(rq4){
    pdf("plots/rq4-1-1.pdf")
  }else if(reverse_warping){
    pdf("plots/rq5-1-1.pdf", width=height, height=height)
    leg <- c("Real Male", "Warped Male", "Adapt Male", "Find Male", "Real Female")
  }else{
    pdf("plots/rq1-1.pdf", width=height, height=height)
  }
}
plot_dens(x=list(dat_test_real_discr, 
                 dat_test_warped_discr,
                 dat_test_adapt_discr,
                 dat_test_find_discr,
                 dat_test_real_non_d),
          leg = leg, 
          xlab = "Amount",
          main = "Real, warped, and FiND world", 
          legend_position = "topright",
          ylim=c(0,0.0003))
if(save_plots){
  dev.off()
}


# 2) similar female distributions in warped and FiND world is supported by a p-value of XXX of
# the respective Kolmogorov-Smirnov test.

# ks.test(dat_test_warped_discr, dat_test_find_discr)
# ks.test(dat_test_adapt_discr, dat_test_find_discr)

# 3) Figure XXX shows the distribution of p-values over all 
# iterations 𝑚 ∈ {1, . . . , 𝑀 } for amount, saving, and risk




chis <- chir <- btr <- bts <- ksr <- kss <- ksa <- vector(length = M_iter)
ksa_a <- bts_a <- btr_a <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  if(reverse_warping){
    dat_test_real_non_d <- list_test_real_female[[m]]
    dat_test_real_discr <- list_test_real_male[[m]]
    dat_test_find_discr <- list_test_find_male[[m]]
    dat_test_warped_discr <- list_test_warped_male[[m]]
    dat_test_adapt_discr <- list_test_adapt_male[[m]]
  }else{
    dat_test_real_non_d <- list_test_real_male[[m]]
    dat_test_real_discr <- list_test_real_female[[m]]
    dat_test_find_discr <- list_test_find_female[[m]]
    dat_test_warped_discr <- list_test_warped_female[[m]]
    dat_test_adapt_discr <- list_test_adapt_female[[m]]
  }
  
  dat_test_real <- rbind(dat_test_real_discr, dat_test_real_non_d)
  dat_test_warped <- rbind(dat_test_warped_discr, dat_test_real_non_d)
  dat_test_adapt <- rbind(dat_test_adapt_discr, dat_test_real_non_d)
  dat_test_find <- rbind(dat_test_find_discr, dat_test_real_non_d)
  
  # Amount
  ksa[m] <- ks.test(dat_test_warped_discr$Amount, dat_test_find_discr$Amount)$p.value
  ksa_a[m] <- ks.test(dat_test_adapt_discr$Amount, dat_test_find_discr$Amount)$p.value
  
  # Saving
  # ks.test(dat_test_warped_discr$Saving, dat_test_find_discr$Saving)
  if(!reverse_warping){
    sav_thresholded <- func_thresh(dat_test_warped, "Saving")
  }else{
    sav_thresholded <- func_thresh(dat_test_warped, "Saving", from="male", to="female")
  }
  kss[m]  <- ks.test(sav_thresholded, dat_test_find_discr$Saving)$p.value
  
  bts[m] <- binom.test(table(sav_thresholded)[2:1], p = mean(dat_test_find_discr$Saving))$p.value
  bts_a[m] <- binom.test(table(dat_test_adapt_discr$Saving)[2:1], p = mean(dat_test_find_discr$Saving))$p.value
  
  chis[m] <- chisq.test(sav_thresholded, dat_test_find_discr$Saving)$p.value  
  
  # Risk
  if(!reverse_warping){
    risk_thresholded <- func_thresh(dat_test_warped, "Risk")
  }else{
    risk_thresholded <- func_thresh(dat_test_warped, "Risk", from="male", to="female")
  }
  ksr[m] <- ks.test(risk_thresholded, dat_test_find_discr$Risk)$p.value  
  
  btr[m] <- binom.test(table(risk_thresholded)[2:1], p = mean(dat_test_find_discr$Risk))$p.value
  btr_a[m] <- binom.test(table(dat_test_adapt_discr$Risk)[2:1], p = mean(dat_test_find_discr$Risk))$p.value
  
  chir[m] <- chisq.test(risk_thresholded, dat_test_find_discr$Risk)$p.value  
}

# Plot 2
if(save_plots){
  if(rq4){
    pdf("plots/rq4-1-2.pdf")
  }else if(reverse_warping){
    pdf("plots/rq5-1-2.pdf")
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

boxplot(ksa, kss, ksr, chis, chir, xlab=NULL, ylab="p-value")
axis(1,at=c(1:5) ,labels=c("amount", "saving ks","risk ks", "saving bin","risk bin"), las=1)
abline(h=0.05)

# Numbers for RQ1
mean(ksa<0.05)
# mean(chis<0.05)
# mean(chir<0.05)
mean(bts<0.05)
mean(btr<0.05)

mean(ksa_a<0.05)
mean(bts_a<0.05)
mean(btr_a<0.05)


#----------------------------------#
#### 2. RQ2 - find most discriminated individuals ####
#----------------------------------#

# "Can individuals be correctly identified that profit or suffer most from 
# unfair discrimination in the real world?"

med_rank <- cor_ranks_base <- cor_ranks_base_a <- wilcox_ranks_base <- cor_ranks <-cor_ranks_a <-  wilcox_ranks <- vector(length = M_iter)
range_rank <- matrix(0,ncol=2, nrow=M_iter)
md <- 50
for(m in seq_len(M_iter)){
  #m <- 1
  dat_eval_map <- list_map[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  
  #   Individuals that profit or suffer most from discrimination: 
  #   Largest gap between real and find world:
  # head(dat_eval_pred[order(dat_eval_pred$diff_f_fr, decreasing=TRUE),])
  # head(dat_eval_pred[order(dat_eval_pred$diff_m_fr, decreasing=TRUE),])
  if(reverse_warping){
    most_discr <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_fr, decreasing=TRUE),]))
  }else{
    most_discr <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_fr, decreasing=TRUE),]))
  }
  
  #   "can be identified..": as most discriminated can be seen those where real
  #     and warped world differs most:
  # head(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=TRUE),])
  if(reverse_warping){
    most_discr_warped <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_wr, decreasing=TRUE),]))
    most_discr_adapt <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_ar, decreasing=TRUE),]))
  }else{
    most_discr_warped <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_wr, decreasing=TRUE),]))
    most_discr_adapt <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_ar, decreasing=TRUE),]))
  }
  
  # Baseline: Just male data instead of male and warped female
  if(reverse_warping){
    most_discr_baseline <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_wr_base, decreasing=TRUE),]))
    most_discr_baseline_adapt <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_m_ar_base, decreasing=TRUE),]))
  }else{
    most_discr_baseline <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_wr_base, decreasing=TRUE),]))
    most_discr_baseline_adapt <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$diff_f_ar_base, decreasing=TRUE),]))
  }
  
  # data frame with IDs and both ranks
  discr_df_find <- data.frame(ID = most_discr, rank_find = seq_len(length(most_discr)))
  
  discr_df_warped <- data.frame(ID = most_discr_warped, rank_warped = seq_len(length(most_discr)))
  discr_df <- merge(discr_df_find, discr_df_warped, by = "ID")

  discr_df_adapt <- data.frame(ID = most_discr_adapt, rank_adapt = seq_len(length(most_discr)))
  discr_df_ad <- merge(discr_df_find, discr_df_adapt, by = "ID")
  
  discr_df_warp_baseline <- data.frame(ID = most_discr_baseline, rank_warped = seq_len(length(most_discr)))
  discr_df_base <- merge(discr_df_find, discr_df_warp_baseline, by = "ID")

  discr_df_adapt_baseline <- data.frame(ID = most_discr_baseline_adapt, rank_adapt = seq_len(length(most_discr)))
  discr_df_adapt_base <- merge(discr_df_find, discr_df_adapt_baseline, by = "ID")
  
    # head(discr_df[order(discr_df$rank_find),])
  # head(discr_df[order(discr_df$rank_warped),])
  
  # Comparison of both vectors:
  #  plot(discr_df$rank_find, discr_df$rank_warped)
  cor_ranks[m] <- cor(discr_df$rank_find, discr_df$rank_warped)
  cor_ranks_a[m] <- cor(discr_df$rank_find, discr_df_ad$rank_adapt)
  #wilcox_ranks[m] <- wilcox.test(discr_df$rank_find, discr_df$rank_warped)$p.value
  
  cor_ranks_base[m] <- cor(discr_df_base$rank_find, discr_df_base$rank_warped)
  cor_ranks_base_a[m] <- cor(discr_df_base$rank_find, discr_df_adapt_base$rank_adapt)
  # wilcox_ranks_base[m] <- wilcox.test(discr_df_base$rank_find, discr_df_base$rank_warped)$p.value
   
  top10 <- most_discr[1:md]
  med_rank[m] <- median(discr_df$rank_warped[top10])
  range_rank[m,] <- range(discr_df$rank_warped[top10])
}

boxplot(med_rank, range_rank[,1], range_rank[,2])
quantile(med_rank)
mean(med_rank<md*2)

#boxplot(wilcox_ranks)
#boxplot(wilcox_ranks_base)



# Plot (2)
if(save_plots){
  if(rq4){
    pdf("plots/rq4-2-2.pdf")
  }else if(reverse_warping){
    pdf("plots/rq5-2-2.pdf")
  }else{
    pdf("plots/rq2-2.pdf")
  }
}
boxplot(cor_ranks, main = "Correlation of ranks of pred diffs", ylab="corr")
if(save_plots){
  dev.off()
}

# Plot für RQ - 3
boxplot(cor_ranks_base, main = "Correlation of ranks of pred diffs - baseline", ylab="corr")
mean(cor_ranks)
quantile(cor_ranks, probs=c(0.025, 0.975))
mean(cor_ranks_a)
quantile(cor_ranks_a, probs=c(0.025, 0.975))
mean(cor_ranks_base)
quantile(cor_ranks_base, probs=c(0.025, 0.975))
mean(cor_ranks_base_a)
quantile(cor_ranks_base_a, probs=c(0.025, 0.975))

t.test(cor_ranks, cor_ranks_base, alternative = "two.sided")
# TODO Hä wieso ist das sogar besser??? Oder muss man sich eh ein anderes Maß anschauen für RQ3?
# (für RQ4-age_sex=TRUE passt es)
# (für RQ5-rev-warping passt es nicht)
# Vielleicht liegt das an diesem Binärisierungsissue https://github.com/slds-lmu/paper_2023_causal_fairness/issues/2 => Ja, vmtl, außerdem an der falschen Baseline: Für prediction nicht warped features nehmen
# Update 13.06.: Na endlich, jetzt passt es für RQ1 :-)
#   .. nicht für das reverse warping.. aber: das ist egal, siehe unten,
#   das generelle Niveau ändert sich nämlich!

#####
# Das mit dem WRS ist auch total Quatsch, die Mediane der differenzen der Ränge sind natürlich 0
# => Test ob H0: warped-real = find-real
# => Sollte nicht abgelehnt werden, also hoher p-Wert
#####

t_wr_total <- t_ar_total <- t_wr_est <- t_wr_est_m <- t_wr_est_total <- t_ar_est_total <- t_wr_base_est <- cor_wr <- cor_wr_base <- t_wr <- t_wr_base <- t_ar <- t_ar_est <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  #m <- 1
  dat_eval_pred <- list_pred[[m]]
  
  wr <- dat_eval_pred$diff_f_wr
  ar <- dat_eval_pred$diff_f_ar
  fr <- dat_eval_pred$diff_f_fr
  wr_base <- dat_eval_pred$diff_f_wr_base
  
  wr_m <- dat_eval_pred$diff_m_wr
  fr_m <- dat_eval_pred$diff_m_fr
  ar_m <- dat_eval_pred$diff_m_ar
  
  wr_total <- c(wr, wr_m)
  fr_total <- c(fr, fr_m)
  ar_total <- c(ar, ar_m)
  
  # plot(wr, fr)
  t_wr[m] <- t.test(wr, fr)$p.value
  t_wr_total[m] <- t.test(wr_total, fr_total)$p.value
  t_wr_est[m] <- diff(t.test(wr, fr)$estimate)
  t_wr_est_m[m] <- diff(t.test(wr_m, fr_m)$estimate)
  t_wr_est_total[m] <- diff(t.test(wr_total, fr_total)$estimate)
  t_ar[m] <- t.test(ar, fr)$p.value
  t_ar_total[m] <- t.test(ar_total, fr_total)$p.value
  t_ar_est[m] <- diff(t.test(ar, fr)$estimate)
  t_ar_est_total[m] <- diff(t.test(ar_total, fr_total)$estimate)
  
    # cor_wr[m] <- cor(wr, fr)
  
  # plot(wr_base, fr)
  t_wr_base[m] <- t.test(wr_base, fr)$p.value
  t_wr_base_est[m] <- diff(t.test(wr_base, fr)$estimate)
  # cor_wr_base[m] <- cor(wr_base, fr)
}

mean(t_wr_est) # => Paper
quantile(t_wr_est, probs=c(0.025,0.975)) # => Paper
mean(t_wr_est_m) 
quantile(t_wr_est_m, probs=c(0.025,0.975)) 
mean(t_wr_est_total) 
quantile(t_wr_est_total, probs=c(0.025,0.975)) 
mean(t_ar_est) # => Paper
quantile(t_ar_est, probs=c(0.025,0.975)) # => Paper
mean(t_ar_est_total) 
quantile(t_ar_est_total, probs=c(0.025,0.975)) 
t.test(t_wr_est) # => wr minimal größer, aber nur 0.1%
t.test(t_wr_est_m) # => wr minimal größer, aber nur 0.1%
t.test(t_wr_base_est) # 4.5% größer

boxplot(t_wr, t_wr_base)
# boxplot(cor_wr, cor_wr_base)
mean(t_wr)
mean(t_wr_base)
# "the null hypothesis of equal differences.."
1-mean(t_wr<0.05)
mean(t_wr_est[t_wr<0.05]) # => Paper
1-mean(t_ar<0.05)
mean(t_ar_est[t_ar<0.05]) # => Paper
1-mean(t_wr_total<0.05)
mean(t_wr_est_total[t_wr_total<0.05]) # => Paper
1-mean(t_ar_total<0.05)
mean(t_ar_est_total[t_ar_total<0.05]) # => Paper

mean(t_wr_base_est[t_wr_base<0.05])
1-mean(t_wr_base<0.05)
mean(t_wr_est)
# .. aber auch hier ist die Baseline nur minimal schlechter besser.. 
# update 13.06.: RQ1: deutlich besser; rev_warping: auch besser
quantile(t_wr, prob=seq(0,1,0.1))
quantile(t_wr, prob=0.25)

md <- 10
top10 <- most_discr[1:md]
top10w <- most_discr_warped[1:md]

bottom10 <- most_discr[(length(most_discr)-md):length(most_discr)]
bottom10w <- most_discr_warped[(length(most_discr)-md):length(most_discr)]

# length(intersect(top10, top10w))

# Plot (1)
if(save_plots){
  if(rq4){
    pdf("plots/rq4-2-1.pdf")
  }else if(reverse_warping){
    pdf("plots/rq5-2-1.pdf")
  }else{
    pdf("plots/rq2-1.pdf", width=height, height=height)
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
          ylab="Difference",
          ylim=c(-0.1, 0.65))
  axis(1,at=c(1:2) ,labels=c("Female Warped-Real", "Female FiND-Real"), las=1)
  # Add hairlines connecting the corresponding values
  for (i in c(top10)){#, bottom10w)) {
    segments(1, dat_eval_pred$diff_f_wr[i], 2, dat_eval_pred$diff_f_fr[i], col = "gray", lty = "solid")
  }
}
if(save_plots){
  dev.off()
}

# Adapt:
if(save_plots){
    pdf("plots/adapt-rq2-1.pdf", width=height, height=height)
}
boxplot(dat_eval_pred$diff_f_ar, dat_eval_pred$diff_f_fr, xlab=NULL, main="Risk prediction differences",
        ylab="Difference",
        ylim=c(-0.1, 0.65))
axis(1,at=c(1:2) ,labels=c("Female Adapt-Real", "Female FiND-Real"), las=1)
# Add hairlines connecting the corresponding values
for (i in c(top10)){#, bottom10w)) {
  segments(1, dat_eval_pred$diff_f_ar[i], 2, 
           dat_eval_pred$diff_f_fr[i], 
           col = "gray", 
           lty = "solid")
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

t_warped_base <- t_warped_est_base <- t_real_est <- t_warped_est <- t_adapt_est <- t_find_est <- t_real <- t_warped <- t_find <- vector(length = M_iter)

for(m in seq_len(M_iter)){
  #m <- 1
  dat_eval_map <- list_map[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  t_real[m] <- t.test(dat_eval_pred$diff_mfr)$p.value
  t_real_est[m] <- t.test(dat_eval_pred$diff_mfr)$estimate
  t_warped[m] <- t.test(dat_eval_pred$diff_mfw)$p.value
  t_warped_est[m] <- t.test(dat_eval_pred$diff_mfw)$estimate
  t_adapt_est[m] <- t.test(dat_eval_pred$diff_mfa)$estimate
  t_find[m] <- t.test(dat_eval_pred$diff_mff)$p.value
  t_find_est[m] <- t.test(dat_eval_pred$diff_mff)$estimate
  
  # TODO comparison with baseline
  mw_base <- dat_eval_pred$pm_real + dat_eval_pred$diff_m_wr_base
  fw_base <- dat_eval_pred$pf_real + dat_eval_pred$diff_f_wr_base
  # #sanity check
  # fw_nonbase <- dat_eval_pred$pf_real + dat_eval_pred$diff_f_wr
  # boxplot(fw_nonbase - dat_eval_pred$pf_warped)
  diff_mfw_base <- mw_base - fw_base
  
  t_warped_base[m] <- t.test(diff_mfw_base)$p.value
  t_warped_est_base[m] <- t.test(diff_mfw_base)$estimate
  
  # => No relevant discrimination in warped world 
  #   (and of course also not in find world, by design)
}

if(save_plots){
  if(rq4){
    pdf("plots/rq4-discr.pdf")
  }else if(reverse_warping){
    pdf("plots/rq5-discr.pdf")
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
  if(rq4){
    pdf("plots/rq4-discr-p.pdf")
  }else if(reverse_warping){
    pdf("plots/rq5-discr-p.pdf")
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
quantile(t_warped, probs = c(0.5,0.69,0.9,0.95,0.97,0.99,1))
# => In about 69%, there would still be a significant difference between
#   risk predictions of male and female subgroup
mean(t_real<0.05)
mean(t_find<0.05)

# Numbers for Paper (RQ1) "The mean difference between risk predictions.."
mean(t_real_est)
quantile(t_real_est, probs=c(0.025,0.975))
mean(t_warped_est)
quantile(t_warped_est, probs=c(0.025,0.975))
mean(t_adapt_est)
quantile(t_adapt_est, probs=c(0.025,0.975))
mean(t_find_est)
quantile(t_find_est, probs=c(0.025,0.975))

t.test(t_real_est)
t.test(t_warped_est) # => Das hier reporten! Das ist der Test, ob im Mittel
# noch eine Diskriminierung vorliegt, tut sie zwar, aber andersrum als vorher 
# und nur sehr klein!
# EDIT 22.03.24: Das hier ist ein Test auf den Mean der Mean differences
# => Warum sollte uns dafür ein CI interessieren? Besser direkt die
#   Quantile oben benutzen
t.test(t_warped_est_base)

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
  
  
  # 
  # boxplot(diff_warped,diff_warped_rev)
  # for (i in seq_len(length(diff_warped))){
  #   segments(1, diff_warped[i], 2, diff_warped_rev[i], col = "gray", lty = "solid")
  # }
  # 
  # boxplot(diff_warped-diff_warped_rev)
  # 
  # plot(density(diff_warped))
  # lines(density(diff_warped_rev))
  # lines(density(diff_warped-diff_warped_rev))
  # head(dat_eval_pred_rev) 
  # head(dat_eval_pred)
  
  ksd[m] <- ks.test(diff_warped, diff_warped_rev)$p.value
}

boxplot(ksd)
quantile(ksd, probs = c(0.25,0.5,0.59,0.7,0.8,0.9))
# => In 58% iterations, KS would reject the null that both warped diffs are 
#   equal, i.e., that reverse_warping DOES make a significant difference in 58% cases
# Update 13.06.: In all the cases?

#########
# Risk ranks
#########


cor_ranks <- cor_ranks_adapt <- wilcox_ranks <- vector(length = M_iter)
pf_warped_all <- pf_warped_all_rev <-pm_warped_all <- pm_warped_all_rev <- vector(length=0)
pf_adapt_all <- pf_adapt_all_rev <-pm_adapt_all <- pm_adapt_all_rev <- vector(length=0)
for(m in seq_len(M_iter)){
  
  dat_eval_map_rev <- list_map_rev[[m]]
  dat_eval_pred_rev <- list_pred_rev[[m]]
  
  dat_eval_map <- list_map[[m]]
  dat_eval_pred <- list_pred[[m]]
  
  pf_warped_id <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_warped, decreasing=TRUE),]))
  pf_warped_id_rev <- as.numeric(rownames(dat_eval_pred_rev[order(dat_eval_pred_rev$pf_warped, decreasing=TRUE),]))
  
  pf_adapt_id <- as.numeric(rownames(dat_eval_pred[order(dat_eval_pred$pf_adapt, decreasing=TRUE),]))
  pf_adapt_id_rev <- as.numeric(rownames(dat_eval_pred_rev[order(dat_eval_pred_rev$pf_adapt, decreasing=TRUE),]))
  
  # data frame with IDs and both ranks
  discr_df_find <- data.frame(ID = pf_warped_id, 
                              rank = seq_len(length(pf_warped_id)))
  
  discr_df_warped <- data.frame(ID = pf_warped_id_rev, 
                                rank_rev = seq_len(length(pf_warped_id_rev)))
  
  discr_df <- merge(discr_df_find, discr_df_warped, by = "ID")
  
  discr_df_find_adapt <- data.frame(ID = pf_adapt_id, 
                              rank = seq_len(length(pf_adapt_id)))
  
  discr_df_adapt <- data.frame(ID = pf_adapt_id_rev, 
                                rank_rev = seq_len(length(pf_adapt_id_rev)))
  
  discr_df_ad <- merge(discr_df_find_adapt, discr_df_adapt, by = "ID")
  
  # Comparison of both vectors:
  cor_ranks[m] <- cor(discr_df$rank, discr_df$rank_rev)
  cor_ranks_adapt[m] <- cor(discr_df_ad$rank, discr_df_ad$rank_rev)
  # wilcox_ranks[m] <- wilcox.test(discr_df$rank, discr_df$rank_rev)$p.value
  
  # Individual predictions
  pf_warped <- dat_eval_pred$pf_warped
  pf_warped_rev <- dat_eval_pred_rev$pf_warped
  ttf[m] <- t.test(pf_warped,pf_warped_rev)$p.value
  
  pf_adapt <- dat_eval_pred$pf_adapt
  pf_adapt_rev <- dat_eval_pred_rev$pf_adapt
  
  pm_warped <- dat_eval_pred$pm_warped
  pm_warped_rev <- dat_eval_pred_rev$pm_warped
  ttm[m] <- t.test(pm_warped,pm_warped_rev)$p.value
  
  pm_adapt <- dat_eval_pred$pm_adapt
  pm_adapt_rev <- dat_eval_pred_rev$pm_adapt
  
  pf_warped_all <- c(pf_warped_all, pf_warped)
  pf_warped_all_rev <- c(pf_warped_all_rev, pf_warped_rev)

  pf_adapt_all <- c(pf_adapt_all, pf_adapt)
  pf_adapt_all_rev <- c(pf_adapt_all_rev, pf_adapt_rev)
  
  pm_warped_all <- c(pm_warped_all, pm_warped)
  pm_warped_all_rev <- c(pm_warped_all_rev, pm_warped_rev)
  
  pm_adapt_all <- c(pm_adapt_all, pm_adapt)
  pm_adapt_all_rev <- c(pm_adapt_all_rev, pm_adapt_rev)
  
}

# boxplot(wilcox_ranks)
boxplot(cor_ranks, main = "Correlation of ranks of pred", ylab="corr")
mean(cor_ranks)
mean(cor_ranks_adapt)
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
  pdf("plots/rq5-general-level-shift.pdf", width=height, height=height)
}
plot_dens(x = list(pf_warped_all, pf_warped_all_rev, pm_warped_all, pm_warped_all_rev),
          leg = c("Female Warped f->m", "Female Warped m->f","Male Warped f->m", "Male Warped m->f"), 
          xlab = "Risk prediction", legend_position = "topleft",
          main = "Risk probs by different warping directions")
if(save_plots){
  dev.off()
}
# .. but the general level is not equal! Warping works in both cases, i.e.,
#   risk predictions are equal in both groups AND individual ranks are equal,
#   only the level changes
if(save_plots){
  pdf("plots/adapt-rq5-general-level-shift.pdf", width=height, height=height)
}
plot_dens(x = list(pf_adapt_all, pf_adapt_all_rev, pm_adapt_all, pm_adapt_all_rev),
          leg = c("Female Adapt f->m", "Female Adapt m->f","Male Adapt f->m", "Male Adapt m->f"), 
          xlab = "Risk prediction", legend_position = "topleft",
          main = "Risk probs by different fairadapt directions")
if(save_plots){
  dev.off()
}
# boxplot(ttf) 
# boxplot(ttm)

# mean(wilcox_ranks<0.05)
mean(ksd<0.05)
mean(ttf<0.05)
mean(ttm<0.05)

#--------------------------------#
#### Classical FairML Metrics ####
#--------------------------------#

for(i in c(1:3)){
  # filenames <- c("results/sim/simulation_2023-08-31_rev_warp_FALSE_age_sex_FALSE_M_1000.RData",
  #                "results/sim/simulation_2023-08-31_rev_warp_FALSE_age_sex_TRUE_M_1000.RData",
  #                "results/sim/simulation_2023-09-01_rev_warp_TRUE_age_sex_FALSE_M_1000.RData")
  filenames <- c(file_name, file_rev, file_rq4)
  cat(filenames[i], "\n")
  load(filenames[i])
  
  fair_mat <- matrix(colMeans(fair_score_mat), byrow = TRUE,
                     ncol=5, dimnames = list(c("real", "warped", "adapt", "find"), c("ACC", "PPV", "FPR", "TPR", "STP")))
  #fair_mat
  
  fair_mat <- cbind(fair_mat, colMeans(fair_check_mat)[1:4])
  fair_mat <- cbind(fair_mat, colMeans(fair_check_mat)[5:8])
  dimnames(fair_mat) <- list(c("real", "warped", "adapt", "find"), c("ACC", "PPV", "FPR", "TPR", "STP", "loss", "n_passes"))
  print(fair_mat)
  print(xtable(fair_mat[,-6], digits=4))
}

# => M_iter=10: Warped überall besser - bei FPR ist am meisten Luft, Richtung ändert sich
# => M_iter=1000: 
#   age_sex FALSE: Warped überall besser - bei FPR ist am meisten Luft, Richtung ändert sich
#   age_sex TRUE:  Warped überall besser außer bei FPR - aber nicht ganz so gut wie oben
#   rev_warp:      Nicht wirklich besser
# TODO: Bei rev_warp und classical metrics muss in der Simu irgendwo noch 
#   ein Fehler sein.



