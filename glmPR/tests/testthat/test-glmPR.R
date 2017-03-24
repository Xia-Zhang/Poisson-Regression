context("running glmPR")

test_that("glmPR stops when meets type error", {
	x <- matrix(1:12, nrow = 4)
	y <- c(1:3)
	expect_error(glmPR(x, y), "Col::subvec(): indices out of bounds or incorrectly used", fixed = TRUE)
	expect_error(glmPR(y ~ x), "variable lengths differ (found for 'x')", fixed = TRUE)
})

test_that("glmPR print function check", {
	x <- matrix(1:12, nrow = 3)
	y <- c(1:3)
	fit = glmPR::glmPR.formula(y ~ x)
	expect_output(str(fit$call), " language glmPR::glmPR.formula(formula = y ~ x)", fixed = TRUE)
})

test_that("glmPR class check", {
	x <- matrix(1:12, nrow = 3)
	y <- c(1:3)
	fit = glmPR::glmPR.formula(y ~ x)
	expect_that(fit, is_a("glmPR"))
})

test_that("glmPR predict stops when meets type error", {
	x <- matrix(1:12, nrow = 3)
	y <- c(1:3)
	fit = glmPR( y ~ x)
	expect_that(predict(fit, newx = c(1:3)), throws_error("on-conformable arguments"))
	expect_that(predict(fit, 1), throws_error("on-conformable arguments"))
})

# test_that("",{
# 	formula = breaks ~ wool+tension, 
# 	data=warpbreaks
# })