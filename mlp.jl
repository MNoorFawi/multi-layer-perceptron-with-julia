
# Multi-layer perceptron MLP
# A multi-layer perceptron, i.e. a fully connected feed-forward deep neural network,
# is basically a bunch of linear regression models stuck together with non-linearities in between.

using Knet, Plots, DataFrames, GZip
# Array{Float32} for cpu and KnetArray{Float32} if Knet.gpu(0) >= 0 for gpu usage
# read gzip data
gz = GZip.open("covtype.data.gz");
data = readtable(gz, separator = ',', header = false); # skipstart argument to skip rows
size(data)
unique(data[end]) , length(unique(data[end]))

# data[.!vec(any(convert(Array, data) .== "?", 2)), :]; to delete rows based on values of any columns

## a function to one-hot encode categorical data and scale numeric data for old and new data based on old
# function preprocess(new::DataFrame, old::DataFrame)
#	dataType = describe(old)
#	x = DataFrame()
#	d = DataFrame()
#	str = dataType[dataType[:eltype] .== String, :variable]
#	num = dataType[(dataType[:eltype] .== Float64) .| (dataType[:eltype] .== Int64), :variable]
#	str = setdiff(str, [names(old)[end]])
#	for i in str
#		dict = unique(old[:, i])
#		for key in dict
#			x[:, [Symbol(key)]] = map(Float32, 1.0(new[:, i] .== key))
#		end
#	end
#	for i in num
#		d[:, i] = map(Float32, (new[:, i]- minimum(new[:, i])) / (maximum(new[:, i]) - minimum(new[:, i])))
#	end
#	x = hcat(x, d)
#	x[:y] = map(UInt8, 1+(new[end] .== "yes")) # to map classes to integers
#	return x
# end;
# encoded_train = preprocess(trn, trn);
# encoded_test = preprocess(tst, trn);
# size(encoded_train), size(encoded_test)

# encode the data with scaling first 10 columns 
x = DataFrame();
for i in 1:size(data, 2)-1
	x[i] = map(Float32, data[i])
	if i <= 10
		x[i] = map(Float32, (data[:, i]- minimum(data[:, i])) / (maximum(data[:, i]) - minimum(data[:, i])))
	end	
end
x[:y] = map(UInt8, data[end]);

# split data randomly into train and test
splits = round(Int, 0.1 * size(x, 1));
shuffled = randperm(size(x, 1));
xtrain, ytrain = [Array(x[shuffled[splits + 1:end], 1:end-1])', Array(x[shuffled[splits + 1:end], end])];
xtest, ytest = [Array(x[shuffled[1:splits], 1:end-1])', Array(x[shuffled[1:splits], end])];

btest = minibatch(xtest, ytest, 100); # [ (x1,y1), (x2,y2), ... ] where xi,yi are minibatches of 100
btrain = minibatch(xtrain, ytrain, 100); # xtype = KnetArray{Float32}
length(btrain), length(btest)

# define predict, loss, and train function of the model
function predict(w, x) #predict(w, x; p = 0) to have a dropout layer
    for i in 1:2:length(w)
        x = w[i] * x .+ w[i+1]
        if i < length(w)-1
            x = max.(0, x)
	    # x = dropout(x, p) the dropout layer
        end
    end
    return x
end;

loss(w, x, yreal) = nll(predict(w, x), yreal)
lossgradient = grad(loss)

function train(model, data, o)
    for (x, y) in data
        grads = lossgradient(model, x, y)
        update!(model, grads, o)
    end
end;

# initial weights (3 layers, 54 inputs => 96 units => 64 units => 32 units => 7 outputs)
w = map(Array{Float32},
	Any[ 0.1f0*randn(96, size(xtrain, 1)), zeros(96, 1),
	     0.1f0*randn(64, 96), zeros(64, 1),
	     0.1f0*randn(32, 64), zeros(32, 1),
             0.1f0*randn(7, 32),  zeros(7, 1) ]);

# define model optimizer
o = optimizers(w, Adam); # o =  optimizers(w, Sgd;  lr=0.01);
		
trnloss = [];
tstloss = [];
trnerror = [];
tsterror = [];

# run the model printing the results, This can take a minute or something because it does so many work in each step
# it calculates accuracy, loss and error for both train and test and they're more than a half a million record
println((:epoch, 0, :train_accuracy, accuracy(w, btrain, predict), :test_accuracy, accuracy(w, btest, predict)))
for epoch in 1:10
    train(w, btrain, o)
	append!(trnloss, nll(w, btrain, predict))
	append!(tstloss, nll(w, btest, predict))
	append!(trnerror, 1-accuracy(w, btrain, predict))
	append!(tsterror, 1-accuracy(w, btest, predict))
    println((:epoch, epoch, :train_accuracy, 1-trnerror[epoch], :test_accuracy, 1-tsterror[epoch]))
    # to calculate only accuracy 	
    # println((:epoch, epoch, :train_accuracy, accuracy(w, btrain, predict), :test_accuracy, accuracy(w, btest, predict)))
end

# plot train loss and test loss
plot([trnloss tstloss], ylim=(0.2, 0.7),
	labels = [:training_loss :test_loss], xlabel = "Epochs", ylabel = "Loss")
# plot train error and test error	
plot([trnerror tsterror], ylim = (0.0, 0.4),
	labels = [:training_error :test_error], xlabel = "Epochs", ylabel = "Error")

# a way to predict new values and check accuracy manually	
yhat = predict(w, xtrain); 
maxval, maxind = findmax(yhat, 1);
yhat = map(x -> ind2sub(yhat, x)[1], maxind);
correct = (map(Float32, yhat[1, :]) .== ytrain);
mean(correct)
yhat

