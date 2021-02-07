% Test dataset
num_seqs = 60000;
num_test = 20000;
seq_size = [1 10 1];

% Create random binary sequences and reshape for network
rng('default');
X = randi([0, 1], [1,10,num_seqs]);
X = reshape(X,10,1,num_seqs);
X = reshape(X, [10,1,1,size(X,3)]);
X = dlarray(X, 'SSCB');

X_Test = randi([0, 1], [1,10,num_test]);
X_Test = reshape(X_Test,10,1,num_test);
X2 = X_Test;
X_Test = reshape(X_Test, [10,1,1,size(X_Test,3)]);
X_Test = dlarray(X_Test, 'SSCB');


latentDim = 5;

encoderLG = layerGraph([imageInputLayer(seq_size,'Name','input_encoder','Normalization','none')
    fullyConnectedLayer(128,'Name', 'fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(128,'Name', 'fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')]);

decoderLG = layerGraph([imageInputLayer([1 1 latentDim],'Name','input_decoder','Normalization','none')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(10, 'Name', 'fc4')
    sigmoidLayer('Name','sm')]);
    
encoder = dlnetwork(encoderLG);
decoder = dlnetwork(decoderLG);


% Training loop
% TRAINING OPTIONS
% Train on auto: use GPU if available.
executionEnvironment = "auto";

%training options for the network
numEpochs = 10;
miniBatchSize = 25;
lr = 0.001;
numIterations = floor(num_seqs/miniBatchSize);
iteration = 0;

%Required for Adam optimizer
avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];

for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations
        iteration = iteration + 1;
        % For each iteration within epoch, obtain next mini-batch from
        % training set
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = X(:,:,:,idx);
        XBatch = dlarray(single(XBatch), 'SSCB');
                    
        % Evaluate the model gradients using dlfeval and modelGradients
        % functions
        % dlfeval: https://uk.mathworks.com/help/deeplearning/ref/dlfeval.html
        % modelGradients: 
        % Update the learnables and average gradients for both networks
        % using adamupdate function
        % adamupdate: https://uk.mathworks.com/help/deeplearning/ref/adamupdate.html
        [infGrad, genGrad] = dlfeval(@modelGradients, encoder, decoder, XBatch);
        
        [decoder.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoder.Learnables, ...
                genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
            
        [encoder.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoder.Learnables, ...
                infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
    end
    elapsedTime = toc; 
    
    % Finally, pass the test set images through autoencoder and calculate
    % loss for this epoch
    [z, zMean, zLogvar] = sample(encoder,X_Test);
    xPred = forward(decoder, z);
    xPred = reshape(xPred,10,1,num_test);
    xPred = dlarray(xPred, 'SSCB');
 
    elbo = vae_loss(X2, xPred, zMean, zLogvar);
    
    % Display epoch number and loss
    disp("Epoch "+epoch+" loss = "+gather(extractdata(elbo))+"("+ elapsedTime + "s)");    
end


function [infGrad, genGrad] = modelGradients(encoder, decoder, x)
    % obtain the encodings obtained from the sampling function.
    [z, z_mu, z_sigma] = sample(encoder, x);

    % obtain predicted dlarray using sigmoid function on the output of the
    % decoder network and input data z
    sz = size(x,4);
    xPred = forward(decoder, z);
    xPred = reshape(xPred,10,1,sz);
    xPred = reshape(xPred, [10,1,1,size(xPred,3)]);
    xPred = dlarray(xPred, 'SSCB');
    
    
    % Calculate the ELBO loss 
    loss = vae_loss(x, xPred, z_mu, z_sigma);
    %loss = dlarray(single(loss), 'SSCB');
    % Calculate the gradients of the loss with respect to the learnable
    % parameters of both encoder and decoder networks.
    [genGrad, infGrad] = dlgradient(loss, decoder.Learnables,encoder.Learnables);
end


function [zSampled, zMean, zLogvar] = sample(encoder, x)
    % x is our sequence batch
    encoded = forward(encoder, x);
    d = size(encoded,1)/2;
    zMean = encoded(1:d,:);
    zLogvar = encoded(1+d:end,:);
    
    % Perform Reparameterization Trick
    sz = size(zMean);
    epsilon = randn(sz);
    sigma = exp(.5 * zLogvar);
    z = epsilon .* sigma + zMean;
    z = reshape(z, [1,1,sz]);

    % Convert encoding to dlarray object in SSCB format
    zSampled = dlarray(z, 'SSCB');
end

function loss = vae_loss(x,xPred,z_mu,z_sigma)
recon_loss = crossentropy(xPred,x,'TargetCategories','independent');
kl_loss = -5e-4 * mean(1 + z_sigma - z_mu.^2 - exp(z_sigma), 1);
loss = mean(recon_loss + kl_loss);
end