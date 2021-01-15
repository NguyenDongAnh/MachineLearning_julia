### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ 3eabb880-32db-11eb-28e3-ed0396b104fa
begin
	using MLDatasets
	using ImageCore
end

# ╔═╡ 1e02bff0-32dd-11eb-3722-09512416ac15
using Flux: onehotbatch

# ╔═╡ f35080fe-32de-11eb-3da2-6d4811d65dc1
using Flux

# ╔═╡ 1b6ff860-32e3-11eb-32a0-87d365df17f5
using Flux: @epochs

# ╔═╡ 7042c0c2-32e3-11eb-3fab-97f0dbe8ce1f
using PlutoUI

# ╔═╡ 96cb4bd0-32e4-11eb-320b-5365b8c42f02
using Statistics

# ╔═╡ f1835ef0-32da-11eb-2939-0b51baa532cb
# Flux for julia
# BigDL for java, Scala
# Pytorch, tensorflow, MXnet, CNTK, etc ... for python

# ╔═╡ 38f94920-32db-11eb-1121-a5a87680baa9
# Today: MNIST with feed-forward ANN (instead of RNN, CNN, ...)

# ╔═╡ 495e8550-32db-11eb-2593-cdf5f803f138
function readTraningData(N)
	A = MNIST.traintensor(Float32, 1:N)
	X = Float32.(zeros(28*28,N))
	for i = 1:N
		X[:,i] = vec(A[:,:,i])
	end
	y = MNIST.trainlabels(1:N) .+ 1
	(X,y)
end

# ╔═╡ 4a118c30-32dc-11eb-0934-8f6ba894a7dd
N = 60000

# ╔═╡ 4e0dc010-32dc-11eb-2fa6-1f4f098740ac
X_train, y_train = readTraningData(N)

# ╔═╡ 740f8462-32dc-11eb-28f9-db9ee78ec9c6
function readTestData(N)
	A = MNIST.testtensor(Float32, 1:N)
	X = Float32.(zeros(28*28,N))
	for i = 1:N
		X[:,i] = vec(A[:,:,i])
	end
	y = MNIST.testlabels(1:N) .+ 1
	(X,y)
end

# ╔═╡ 9f8d8ce0-32dc-11eb-3815-81030bfa7be6
M = 10000

# ╔═╡ c321faae-32dc-11eb-264f-63f38ad2bde3
X_test , y_test = readTestData(M)

# ╔═╡ d20d4110-32dc-11eb-1194-d95f420d2f30
first_x = MNIST.convert2image(MNIST.traintensor(1))

# ╔═╡ 0d54e200-32dd-11eb-1fcb-65f63c424674
first_y = y_train[1]

# ╔═╡ 19026280-32dd-11eb-1f56-4fda7c01341d
y_train

# ╔═╡ 9353dc82-32dd-11eb-3e36-67ca02f9fd37
Y_train = onehotbatch(y_train,1:10)

# ╔═╡ ad916400-32dd-11eb-345e-fd992854a000
# X_train: matrix of size 784xN
# Y_train: label matrix of size 10xN

# ╔═╡ 2848cbc2-32de-11eb-1785-ab3fd60c7895
model = Chain(Dense(784,256, σ),Dense(256,128,σ),Dense(128,10),softmax)

# ╔═╡ 1f44caf0-32df-11eb-2c5d-43c830089a76
θ = Flux.params(model)

# ╔═╡ 28e58b80-32df-11eb-2124-23757949d176
θ[4]

# ╔═╡ f947e8a2-32de-11eb-0a60-8f011e71b234
first_prediction = model(X_train[:,1])

# ╔═╡ 10c01920-32e0-11eb-246a-d37551856fcd
Flux.onecold(first_prediction)

# ╔═╡ ee5282a0-32e0-11eb-21ed-07b3e848de25
loss(X,y) = Flux.crossentropy(model(X),y)

# ╔═╡ 0ce47020-32e1-11eb-2557-e901d18092a8
initialLoss = loss(X_train, Y_train)

# ╔═╡ 7b38c300-32e1-11eb-34c0-3357fd62e4d5
optimizer = ADAM()

# ╔═╡ ec605a20-32e1-11eb-24f0-7396f4daebf7
Flux.train!(loss, θ,[(X_train, Y_train)],optimizer)

# ╔═╡ 48029aa0-32e2-11eb-2fad-a792beed3f97
θ[4]

# ╔═╡ 2503e7b0-32e3-11eb-2578-e594edc54ee5
function train(numEpochs=20)
	with_terminal() do
		@epochs numEpochs Flux.train!(loss, θ,[(X_train, Y_train)],optimizer; cb = ()-> println(loss(X_train, Y_train)))
	end
end

# ╔═╡ 451f5b60-32e3-11eb-02cc-d18cab7b9fcf
train(10)

# ╔═╡ 1db956b2-32e4-11eb-12ff-d3f67246ba45
initialLoss

# ╔═╡ 372c1420-32e4-11eb-2862-f75d270ceed2
accuracy(X,y) = mean(Flux.onecold(model(X)) .== y)

# ╔═╡ 6d79c040-32e4-11eb-35c7-a56f0357d42b
training_accuracy = accuracy(X_train, y_train)

# ╔═╡ af08e4a0-32e4-11eb-0a8c-fbd53b12f814
test_accuracy = accuracy(X_test, y_test)

# ╔═╡ 15381de2-4843-11eb-06ad-b55d7129b95e
model(X_train)

# ╔═╡ Cell order:
# ╠═f1835ef0-32da-11eb-2939-0b51baa532cb
# ╠═38f94920-32db-11eb-1121-a5a87680baa9
# ╠═3eabb880-32db-11eb-28e3-ed0396b104fa
# ╠═495e8550-32db-11eb-2593-cdf5f803f138
# ╠═4a118c30-32dc-11eb-0934-8f6ba894a7dd
# ╠═4e0dc010-32dc-11eb-2fa6-1f4f098740ac
# ╠═740f8462-32dc-11eb-28f9-db9ee78ec9c6
# ╠═9f8d8ce0-32dc-11eb-3815-81030bfa7be6
# ╠═c321faae-32dc-11eb-264f-63f38ad2bde3
# ╠═d20d4110-32dc-11eb-1194-d95f420d2f30
# ╠═0d54e200-32dd-11eb-1fcb-65f63c424674
# ╠═1e02bff0-32dd-11eb-3722-09512416ac15
# ╠═19026280-32dd-11eb-1f56-4fda7c01341d
# ╠═9353dc82-32dd-11eb-3e36-67ca02f9fd37
# ╠═ad916400-32dd-11eb-345e-fd992854a000
# ╠═f35080fe-32de-11eb-3da2-6d4811d65dc1
# ╠═2848cbc2-32de-11eb-1785-ab3fd60c7895
# ╠═1f44caf0-32df-11eb-2c5d-43c830089a76
# ╠═28e58b80-32df-11eb-2124-23757949d176
# ╠═f947e8a2-32de-11eb-0a60-8f011e71b234
# ╠═10c01920-32e0-11eb-246a-d37551856fcd
# ╠═ee5282a0-32e0-11eb-21ed-07b3e848de25
# ╠═0ce47020-32e1-11eb-2557-e901d18092a8
# ╠═7b38c300-32e1-11eb-34c0-3357fd62e4d5
# ╠═ec605a20-32e1-11eb-24f0-7396f4daebf7
# ╠═48029aa0-32e2-11eb-2fad-a792beed3f97
# ╠═1b6ff860-32e3-11eb-32a0-87d365df17f5
# ╠═7042c0c2-32e3-11eb-3fab-97f0dbe8ce1f
# ╠═2503e7b0-32e3-11eb-2578-e594edc54ee5
# ╠═451f5b60-32e3-11eb-02cc-d18cab7b9fcf
# ╠═1db956b2-32e4-11eb-12ff-d3f67246ba45
# ╠═96cb4bd0-32e4-11eb-320b-5365b8c42f02
# ╠═372c1420-32e4-11eb-2862-f75d270ceed2
# ╠═6d79c040-32e4-11eb-35c7-a56f0357d42b
# ╠═af08e4a0-32e4-11eb-0a8c-fbd53b12f814
# ╠═15381de2-4843-11eb-06ad-b55d7129b95e
