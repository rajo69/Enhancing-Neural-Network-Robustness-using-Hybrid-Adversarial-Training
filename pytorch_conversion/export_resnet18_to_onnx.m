clc;
clear;

disp("=== MATLAB → ONNX Export Script ===");

%% ===== CONFIGURATION =====

% List ALL model files here
modelFiles = [
    "resnet_18_v1.mat"
    "resnet_18_v2.mat"
    "resnet_18_v3.mat"
];

% ONNX Opset version (11 is safest for PyTorch)
OPSET = 11;

%% ===== PROCESS EACH MODEL =====

for i = 1:length(modelFiles)

    fprintf("\nProcessing: %s\n", modelFiles(i));

    %% ---- Load MAT file ----
    data = load(modelFiles(i));

    % Detect network variable automatically
    netVar = fieldnames(data);
    net = data.(netVar{1});

    %% ---- Convert to supported format ----
    % Handles DAGNetwork, SeriesNetwork, and dlnetwork

    if isa(net, "SeriesNetwork") || isa(net, "DAGNetwork")
        disp("Network type: Series/DAG Network (OK)");
        
    elseif isa(net, "dlnetwork")
        disp("Network type: dlnetwork → converting to layerGraph");
        net = layerGraph(net);

    else
        error("Unsupported network type!");
    end

    %% ---- Fix common ONNX export issues ----
    % Convert to layerGraph to avoid unsupported layer errors
    try
        lgraph = layerGraph(net);
    catch
        lgraph = net;
    end

    %% ---- Define dummy input (IMPORTANT!) ----
    % CIFAR-10 ResNet18 typically uses 32x32x3 input
    
    inputSize = [32 32 3];
    
    dummyInput = rand(inputSize, "single");

    %% ---- Generate ONNX filename ----
    [~, name, ~] = fileparts(modelFiles(i));
    onnxFile = name + ".onnx";

    %% ---- Export to ONNX ----
    try
        exportONNXNetwork(lgraph, onnxFile, ...
            "OpsetVersion", OPSET, ...
            "InputData", dummyInput);

        fprintf("✅ Successfully exported: %s\n", onnxFile);

    catch ME

        warning("Standard export failed. Trying fallback method...");
        
        % Fallback export method
        exportONNXNetwork(net, onnxFile, ...
            "OpsetVersion", OPSET, ...
            "InputData", dummyInput);

        fprintf("✅ Exported using fallback: %s\n", onnxFile);
    end

end

disp("=== ALL MODELS EXPORTED SUCCESSFULLY ===");
