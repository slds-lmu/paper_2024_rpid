# Plotting function 
plot_dens <- function(x,
                      leg = c("name of x1", "name of x2", "name of x3"),
                      xlab = "xlab",
                      main = "Density of foo",
                      single_panel = TRUE,
                      plot_legend = TRUE,
                      legend_position = "topleft",
                      fn = density,
                      ...) {
  if(plot_legend){if(length(x)!=length(leg)){warning("Number of vectors and length of legend does not match")}}
  if(single_panel){
    par(mfrow = c(1, 1))
  }
  plot(fn(x[[1]]),
       lwd = 2,
       xlab = xlab,
       main = main,
       ...)
  if(length(x)>1){
    lines(fn(x[[2]]),
          col = "blue",
          lty = 2,
          lwd = 2)
    if (length(x) > 2){
      lines(fn(x[[3]]),
            col = "orange",
            lty = 3,
            lwd = 2)
      if (length(x) > 3) {
        lines(fn(x[[4]]),
              col = "gray",
              lty = 4,
              lwd = 2)
        if (length(x) > 4) {
          lines(fn(x[[5]]),
                col = "green",
                lty = 5,
                lwd = 2)
          if(plot_legend){legend(
            legend_position,
            legend = c(leg[1], leg[2], leg[3], leg[4], leg[5]),
            lty = c(1, 2, 3, 4, 5),
            col = c("black", "blue", "orange", "gray", "green"),
            lwd = 2
          )}
        }else{
          if(plot_legend){legend(
            legend_position,
            legend = c(leg[1], leg[2], leg[3], leg[4]),
            lty = c(1, 2, 3, 4),
            col = c("black", "blue", "orange", "gray"),
            lwd = 2
          )}
        }
      }else{
        if(plot_legend){legend(
          legend_position,
          legend = c(leg[1], leg[2], leg[3]),
          lty = c(1, 2, 3),
          col = c("black", "blue", "orange"),
          lwd = 2
        )}
      }
    }else{
      if(plot_legend){legend(
        legend_position,
        legend = c(leg[1], leg[2]),
        lty = c(1, 2),
        col = c("black", "blue"),
        lwd = 2
      )}
    }
  }
}

# Draw random numbers from a gamma distribution where parameters are estimated
#   first from a given vector
random_gamma <- function(x,
                         n,
                         var_name = "",
                         verbose = TRUE) {
  sigma <- var(x) / mean(x)
  alpha <- mean(x) / sigma
  y <- rgamma(n, shape = alpha, scale = sigma)
  if (verbose) {
    cat(
      var_name,
      "-- Expectation:",
      alpha * sigma,
      "-- Variance:",
      alpha * sigma ^ 2,
      "-- Range:",
      range(y),
      "-- n:",
      length(y),      
      "\n"
    )
  }
  return(y)
}

sigmoid <- function(eta) {
  exp(eta) / (1 + exp(eta))
}

check_probs <- function(x){
  if (min(x) < 0 |
      max(x) > 1) {
    stop("probs outside (0,1)")
  }else{
    print(range(x))}
}


##########################
# Warping function
##########################
warp <- function(x, # vector, source distribution - old quantiles
                 y, # vector, target distribution - new quantiles
                 to_warp = NULL) { # vector, to be warped from old to new
  if (is.null(to_warp)) {
    to_warp <- x
  }
  warped <- vector(length = length(to_warp))
  x_sort <- sort(x)
  e_cdf <- 1:length(x_sort) / length(x_sort)
  x_e_cdf <- data.frame(x_sort, e_cdf)
  
  # for each old value...
  for (i in seq_len(length(to_warp))) {
    old_value <- to_warp[i]
    # ...find the p to this quantile in the old vector
    my_q <-
      x_e_cdf[which(x_sort >= old_value)[1], "e_cdf"]
    if(is.na(my_q)){my_q <- 1} # set to max if out of range
    # ... warp to the p-quantile in the target distribution
    warped[i] <- quantile(y, my_q)
  }
  return(warped)
}

warp_model <- function(dat_x,
                       model_x,
                       model_y
){
  warped_res <- warp(x = residuals(model_x, type="response"), 
                     y = residuals(model_y, type="response"))
  warped_pred <- predict(model_y, newdata = dat_x, type = "response") + warped_res
  return(warped_pred)
}

# Simulate new person
sim_new <- function(age_new, mod_sav_age, mod_am_age, Sex){
  p_sav_new <- predict(mod_sav_age, newdata = data.frame(Age=age_new, Sex=Sex))
  Sav_new <- rbinom(1, 1, prob = p_sav_new)
  am_mean_new <- predict(mod_am_age, newdata=data.frame(Age=age_new, Sex=Sex))
  var_am <- var(mod_am_age$residuals)
  sigma <- var_am / am_mean_new
  alpha <- am_mean_new / sigma
  Am_new <- rgamma(1, shape = alpha, scale = sigma)
  dat_new <- data.frame(Age = age_new,
                        Saving = Sav_new,
                        Amount = Am_new,
                        Sex=Sex)
  return(dat_new)
}

# a) Map / Warp data point to FiND world
warp_new_data <- function(dat_new, model_x, model_y, target=NULL){
  pred_new <- predict(model_x, newdata = dat_new, type="response")
  res_new <- as.numeric(dat_new[,target] - pred_new)
  warped_res <- warp(x = residuals(model_x, type="response"), 
                     y = residuals(model_y, type="response"),
                     to_warp = res_new)
  warped_pred <- predict(model_y, newdata = dat_new, type = "response") + warped_res
  return(warped_pred)
}

normalize <- function(x){
  x_norm <- (x-mean(x))/sd(x)
  return(x_norm)}

normalize_df <- function(dat, cols){
  dat_n <- dat
  for(col in cols){
    dat_n[,col] <- normalize(dat[,col])
  }
  return(dat_n)
}

plot_warped <- function(dat_female_warped, dat_female, target, ...){
  plot_dens(x=list(dat_female_warped[,target], 
                   dat_female[,target]),
            leg = c("Warped Female", "Real Female"), 
            xlab = target,
            main = "Comparison real and warped world",legend_position = "topright",
            ...)
}

plot_pred <- function(dat, cols, ...){
  plot_dens(x=list(dat[,cols[1]], 
                   dat[,cols[2]]),
            leg = cols, 
#            xlab = target,
            main = "Comparison predictions",legend_position = "topright",
            ...)
}



# Compute MSEs between corresponding columns in 2 data sets
mse_func_col <- function(dat_female_warped, dat_female, cols){
  x <- vector(length=length(cols))
  dfw_n <- normalize_df(dat_female_warped, cols=cols)
  df_n <- normalize_df(dat_female, cols=cols)
  for(i in seq_len(length(cols))){
    x[i] <- mean((dfw_n[,cols[i]] - df_n[,cols[i]])^2)
  }
  names(x) <- cols
  return(x)
}

# Compute MSEs between corresponding rows in 2 data sets
mse_func_row <- function(dat_female_warped, dat_female, cols){
  x <- vector(length=nrow(dat_female_warped))
  dfw_n <- normalize_df(dat_female_warped, cols=cols)
  df_n <- normalize_df(dat_female, cols=cols)
  for(i in seq_len(length(x))){
    x[i] <- mean(as.matrix((dfw_n[i,cols] - df_n[i,cols])^2))
  }
  return(x)
}

# Threshold female numeric scores to binary values 
# wrt marginal distribution of males
func_thresh <- function(data, col, from="female", to="male"){
  qm <- 1 - mean(data[,col][data$Sex==to])
  col_fem <- data[,col][data$Sex==from]
  col_thresholded_f <- ifelse(col_fem<quantile(col_fem, qm),0,1)
  return(col_thresholded_f)
}

# Round risk values of female scores to binary values 
# wrt marginal distribution of males
round_risk <- function(risk_star_f, risk_m){
  qm <- 1 - mean(risk_m)
  risk_thresholded <- ifelse(risk_star_f<quantile(risk_star_f, qm),0,1)
  return(factor(risk_thresholded, levels = c(0,1), labels=c("bad", "good")))
}