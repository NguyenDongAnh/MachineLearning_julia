### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ c0af7de0-2d67-11eb-0dde-454dd0862444
using Flux

# ╔═╡ 4206f3e0-2d69-11eb-21d9-15de3cad1628
using Flux: onehot

# ╔═╡ a6c0c7c0-2d69-11eb-1bb5-cde87d909f04
using Flux: onehotbatch

# ╔═╡ 755d6390-2d6a-11eb-31cd-a79c45c6d608
using Flux: crossentropy

# ╔═╡ d0bd9960-2d67-11eb-0e3f-a59a1cd4ba75
X = Flux.Data.Iris.features()

# ╔═╡ e2adbb50-2d67-11eb-0424-bd3ccdfcd55f
labels = Flux.Data.Iris.labels()

# ╔═╡ 2fe35b50-2d68-11eb-3a0e-b78ac5a2809d
unique(labels)

# ╔═╡ 36f6e150-2d68-11eb-123f-17eeb627eacf
indexin([:a :b :c :a :c],[:a :b :c])

# ╔═╡ 9a63e9e0-2d68-11eb-0a43-6f45744c6374
z= indexin(labels,unique(labels))

# ╔═╡ b36547e0-2d68-11eb-05cd-795a0ccf914c
# 1=> [1 0 0]; 2 =>[0 1 0]; 3 => [0 0 1]

# ╔═╡ 486e5b60-2d69-11eb-067a-0f949aaa6653
onehot(:b,[:a,:b,:c])

# ╔═╡ 6366a762-2d69-11eb-0b3b-67df6d2479ca
onehotbatch([:b,:c],[:a,:b,:c])

# ╔═╡ a3d874e0-2d69-11eb-3ea7-e9ae71e9a092
y = onehotbatch(z,unique(1:3))

# ╔═╡ c7ba3f60-2d69-11eb-3e97-fbb5bf9f6b91
#model = Chain(Dense(4,3),softmax) #MLR model
model = Chain(Dense(4,3,σ),Dense(3,3),softmax) #MLR model

# ╔═╡ a0f4f970-4843-11eb-30c9-9fa2430b0f24
model(X)

# ╔═╡ 44693160-2d6a-11eb-18a1-0f3513ba8f4d
loss(x,y) = Flux.crossentropy(model(x),y)

# ╔═╡ bb3fd1f0-2d69-11eb-1abf-1519f52c0106
θ = Flux.params(model)

# ╔═╡ b2e138e0-2d6a-11eb-0b1e-652d0fe61d7e
θ[1]

# ╔═╡ c50255e0-2d6a-11eb-0310-db389258bf65
θ[2]

# ╔═╡ caaef8de-2d6a-11eb-133d-8b426176caf8
θ[3]

# ╔═╡ 077a30c0-483f-11eb-0aec-0be69c89b3a6
θ[4]

# ╔═╡ d0f6a360-2d6a-11eb-0d08-6f8ffecce416
loss(X,y)

# ╔═╡ 923d6a12-483f-11eb-3b9f-73eae01fb7f3
optimizer = ADAM()

# ╔═╡ 8ee4c930-483f-11eb-182c-5be574ff127c
Flux.train!(loss, θ,[(X, y)],optimizer)

# ╔═╡ 0ab73de2-4840-11eb-05cb-2b80b512c7f9
# using Flux: @epochs

# ╔═╡ 0f7429b0-4840-11eb-0af7-c95897ee540a
# using PlutoUI

# ╔═╡ bad45830-483f-11eb-28e7-7b916a959762
# function train(numEpochs=20)
# 	with_terminal() do
# 		@epochs numEpochs Flux.train!(loss, θ,[(X, y)],optimizer; cb = ()-> println(loss(X, y)))
# 	end
# end

# ╔═╡ 18ea7ee0-4840-11eb-255b-bb51d9e09f1f
# train(10000)

# ╔═╡ 54d73970-4840-11eb-1b29-3138bb3539f7
# using Statistics

# ╔═╡ 6325a390-4840-11eb-2ea4-0ff0cfda243f
# accuracy(X,y) = mean(Flux.onecold(model(X)) .== y)

# ╔═╡ 68f0f220-4840-11eb-085d-73f2820144bc
# model(X)

# ╔═╡ 0a1bf600-4845-11eb-359a-5bb8d77b612a
# θ[4]

# ╔═╡ 2e1922a2-4843-11eb-38b7-8194794adb86
# accuracy(X,z)

# ╔═╡ d59f43de-2d6a-11eb-37f1-1de870dd3444
#Next step: Compute gradient of the loss function with respect to θ
#Update parameters using gradient descent or quasi-Newton algorithm
#Read documentation of Flux

# ╔═╡ a3e0bffe-4849-11eb-2454-9b317970281e


# ╔═╡ ce44f410-4849-11eb-0abd-b362d8ab32a8
begin
	m = Chain(Dense(10, 5), Dense(5, 2));
	x = [0.0809912 0.417673 0.412175 0.8976 0.939981 0.781591 0.000997935 0.71329 0.109329 0.422887]
	m(x')
	θ1 = Flux.params(m)
end

# ╔═╡ Cell order:
# ╠═c0af7de0-2d67-11eb-0dde-454dd0862444
# ╠═d0bd9960-2d67-11eb-0e3f-a59a1cd4ba75
# ╠═e2adbb50-2d67-11eb-0424-bd3ccdfcd55f
# ╠═2fe35b50-2d68-11eb-3a0e-b78ac5a2809d
# ╠═36f6e150-2d68-11eb-123f-17eeb627eacf
# ╠═9a63e9e0-2d68-11eb-0a43-6f45744c6374
# ╠═b36547e0-2d68-11eb-05cd-795a0ccf914c
# ╠═4206f3e0-2d69-11eb-21d9-15de3cad1628
# ╠═486e5b60-2d69-11eb-067a-0f949aaa6653
# ╠═a6c0c7c0-2d69-11eb-1bb5-cde87d909f04
# ╠═6366a762-2d69-11eb-0b3b-67df6d2479ca
# ╠═a3d874e0-2d69-11eb-3ea7-e9ae71e9a092
# ╠═c7ba3f60-2d69-11eb-3e97-fbb5bf9f6b91
# ╠═755d6390-2d6a-11eb-31cd-a79c45c6d608
# ╠═a0f4f970-4843-11eb-30c9-9fa2430b0f24
# ╠═44693160-2d6a-11eb-18a1-0f3513ba8f4d
# ╠═bb3fd1f0-2d69-11eb-1abf-1519f52c0106
# ╠═b2e138e0-2d6a-11eb-0b1e-652d0fe61d7e
# ╠═c50255e0-2d6a-11eb-0310-db389258bf65
# ╠═caaef8de-2d6a-11eb-133d-8b426176caf8
# ╠═077a30c0-483f-11eb-0aec-0be69c89b3a6
# ╠═d0f6a360-2d6a-11eb-0d08-6f8ffecce416
# ╠═923d6a12-483f-11eb-3b9f-73eae01fb7f3
# ╠═8ee4c930-483f-11eb-182c-5be574ff127c
# ╠═0ab73de2-4840-11eb-05cb-2b80b512c7f9
# ╠═0f7429b0-4840-11eb-0af7-c95897ee540a
# ╠═bad45830-483f-11eb-28e7-7b916a959762
# ╠═18ea7ee0-4840-11eb-255b-bb51d9e09f1f
# ╠═54d73970-4840-11eb-1b29-3138bb3539f7
# ╠═6325a390-4840-11eb-2ea4-0ff0cfda243f
# ╠═68f0f220-4840-11eb-085d-73f2820144bc
# ╠═0a1bf600-4845-11eb-359a-5bb8d77b612a
# ╠═2e1922a2-4843-11eb-38b7-8194794adb86
# ╠═d59f43de-2d6a-11eb-37f1-1de870dd3444
# ╠═a3e0bffe-4849-11eb-2454-9b317970281e
# ╠═ce44f410-4849-11eb-0abd-b362d8ab32a8
