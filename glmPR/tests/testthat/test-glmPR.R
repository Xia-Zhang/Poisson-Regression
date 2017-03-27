context("running glmPR")

test_that("glmPR stops when meets type error", {
	x <- matrix(rnorm(100), ncol = 4)
	y <- rpois(10, 3)
	expect_error(glmPR(x, y), "y_nrow == x_nrow is not TRUE", fixed = TRUE)
	expect_error(glmPR(y ~ x), "variable lengths differ (found for 'x')", fixed = TRUE)
	x <- rnorm(1000)
	y <- 1
	expect_error(glmPR(x, y), "y_nrow == x_nrow is not TRUE", fixed = TRUE)
	expect_error(glmPR(y ~ x), "variable lengths differ (found for 'x')", fixed = TRUE)
})

test_that("glmPR print function check", {
	x <- matrix(rnorm(100), ncol = 4)
	y <- rpois(25, 3)
	fit = glmPR(y ~ x)
	expect_output(str(fit$call), " language glmPR.formula(formula = y ~ x", fixed = TRUE)
})

test_that("glmPR class check", {
	x <- matrix(rnorm(100), ncol = 4)
	y <- rpois(25, 3)
	fit = glmPR(y ~ x)
	expect_that(fit, is_a("glmPR"))
})

test_that("glmPR predict stops when meets type error", {
	x <- matrix(rnorm(100), ncol = 4)
	y <- rpois(25, 3)
	fit = glmPR( y ~ x)
	expect_that(predict(fit, newx = c(1:3)), throws_error("on-conformable arguments"))
	expect_that(predict(fit, 1), throws_error("on-conformable arguments"))
})

test_that("glmPR stops when thread number is less than 1",{
	x <- matrix(rnorm(100), ncol = 4)
	y <- rpois(25, 3)
	expect_error(glmPR(x, y, 0.1, 0), "the thread number shouldn't be 0 or less")
})

test_that("glmPR stops when memory allocation failed", {
	x <- matrix(rnorm(1000), nrow = 2)
	y <- c(1, 2)
	expect_error(glmPR(x, y, 0.1, 1e6), "memory allocation failed")
})