""" Componets of the model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

__all__ = ['GCN_E2_decline_L2_log_clf1_selu_multi', 'GCN_E2_decline_L2_div10_clf1_selu_multi', 'GCN_E2_decline_L2_sqr_clf1_selu_multi', 'GCN_E2_decline_L3_sqr_clf1_selu_multi', 'GCN_E2_decline_L4_sqr_clf1_selu_multi', 'GCN_E2_decline_L5_sqr_clf1_selu_multi', 'SimpleNN_relu']

def xaviern_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, 1)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
        
def xavieru_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, 1)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

def knorm_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)
        
def kuni_init(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)   

class Linear_init():
    def __init__(self, act_func):
        self.act_func = act_func
        
    def select_init(m):
        if type(m) == nn.Linear:
            if (self.act_func == "leaky_relu_xn") | (self.act_func == "selu_xn"):
                nn.init.xavier_normal_(m.weight, 1)
            elif (self.act_func == "leaky_relu_xu") | (self.act_func == "selu_xu"):
                nn.init.xavier_uniform_(m.weight, 1)
            elif self.act_func == "kn":
                nn.init.kaiming_normal_(m.weight)
            elif self.act_func == "ku":          
                nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
               m.bias.data.fill_(0.0)        

def select_init(m):
    if type(m) == nn.Linear:
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            nn.init.xavier_normal_(m.weight, 1)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            nn.init.xavier_uniform_(m.weight, 1)
        elif act_func == "kn":
            nn.init.kaiming_normal_(m.weight)
        elif act_func == "ku":          
            nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)    
        

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, act_func, dropout, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        if self.bias is not None:
            self.bias.data.fill_(0.0)
        self.dropout = UniformDropout(dropout)
        self.act = nn.SELU()

        if act_func == "selu_xn":
            nn.init.xavier_normal_(self.weight.data, 3/4)
        elif act_func == "selu_xu":
            nn.init.xavier_uniform_(self.weight.data, 3/4)
        elif (act_func == "kn") | (act_func == "leaky_relu_xn"):
            nn.init.kaiming_normal_(self.weight.data)
        elif (act_func == "ku") | (act_func == "leaky_relu_xu"):    
            nn.init.kaiming_uniform_(self.weight.data)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        output = self.dropout(output)
        output = self.act(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output    

    
class UniformDropout(nn.Module):
    """
    https://arxiv.org/pdf/1801.05134.pdf
    """
    def __init__(self, p):
        super(UniformDropout, self).__init__()
        self.beta = p

    def forward(self, x):
        if self.training:
            noise = torch.rand(*x.size()).to(x.device) * self.beta * 2 - self.beta
            return x * (1 + noise)
        else:
            return x   


class SimpleNN_relu(nn.Module):
    def __init__(self, in_dim, out_dim, _, __):
        super(SimpleNN_relu, self).__init__()
        self.out_dim = out_dim  
        
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(in_dim, 50)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 5)
        
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)        
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(5, num_class)))
        #self.fc5 = nn.Linear(5, out_dim)

    def forward(self, x, _):
        out = []
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        #x = F.sigmoid(self.fc5(x))
        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out    

class ModuleWithHooks(nn.Module):
    def __init__(self, base_module):
        super().__init__()
        self.base_module = base_module
        self.save_input = SaveInput()

        self.base_module.register_forward_hook(self.save_input.save_input)

    def forward(self, *args, **kwargs):
        return self.base_module(*args, **kwargs)

# Define a new class to save inputs
class SaveInput:
    def __init__(self):
        self.input_tensor = None

    def save_input(self, module, input_tensor, output):
        print(f"Saving input for module: {module}")  # debug statement
        self.input_tensor = input_tensor

def get_model_with_hooks(base_model_class_name):
    base_model_class = base_model_class_name #models.__dict__[base_model_class_name]
    print("base_model_class_name", base_model_class_name)
    
    class ModelWithHooks(base_model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Find all graph convolution layers and classifiers
            self.gcs = [module for name, module in self.named_modules() if name.startswith('gc')]
            self.linears = [module for name, module in self.named_modules() if isinstance(module, torch.nn.Linear)]

            # Register hooks
            self.save_inputs = {module: SaveInput() for module in self.gcs + self.linears}
            for module, save_input in self.save_inputs.items():
                module.register_forward_hook(save_input.save_input)

    return ModelWithHooks

class GCN_E2_decline_L3_sqr_clf1_selu_multi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__()
        in_dim2 = int(math.sqrt(in_dim))

        if act_func == "xn":
            act_func = "selu_xn"
        elif act_func == "xu":
            act_func = "selu_xu"
        self.out_dim = out_dim    
        
        self.gc1 = GraphConvolution(in_dim, in_dim2, act_func, dropout) 
        self.gc2 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.gc3 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(in_dim2, num_class)))
                            
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            for i in range(len(out_dim)):
                self.clf[i].apply(xaviern_init)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            self.clf1.apply(xavieru_init)
            self.clf2.apply(xavieru_init)
        elif act_func == "kn":
            self.clf1.apply(knorm_init)
            self.clf2.apply(knorm_init)
        elif act_func == "ku":          
            self.clf1.apply(kuni_init)  
            self.clf2.apply(kuni_init)  
        
    def forward(self, x, adj):
        out = []
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)

        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out        

# Modify your model to save inputs
class lrpModel(GCN_E2_decline_L3_sqr_clf1_selu_multi):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__(in_dim, out_dim, dropout, act_func)

        # Create an instance of SaveInput for each layer
        self.save_input_gc1 = SaveInput()
        self.save_input_gc2 = SaveInput()
        self.save_input_gc3 = SaveInput()
        self.save_input_clfs = [SaveInput() for _ in out_dim]

        # Register hooks
        self.gc1.register_forward_hook(self.save_input_gc1.save_input)
        self.gc2.register_forward_hook(self.save_input_gc2.save_input)
        self.gc3.register_forward_hook(self.save_input_gc3.save_input)
        for clf, save_input_clf in zip(self.clf, self.save_input_clfs):
            clf.register_forward_hook(save_input_clf.save_input)

    def get_saved_inputs(self):
        saved_inputs = [self.save_input_gc1.input_tensor, 
                        self.save_input_gc2.input_tensor, 
                        self.save_input_gc3.input_tensor] + \
                        [save_input_clf.input_tensor for save_input_clf in self.save_input_clfs]
        return saved_inputs
    
    
class GCN_E2_decline_L2_sqr_clf1_selu_multi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__()
        in_dim2 = int(math.sqrt(in_dim))

        if act_func == "xn":
            act_func = "selu_xn"
        elif act_func == "xu":
            act_func = "selu_xu"
        self.out_dim = out_dim    
        
        self.gc1 = GraphConvolution(in_dim, in_dim2, act_func, dropout) 
        self.gc2 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(in_dim2, num_class)))
                            
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            for i in range(len(out_dim)):
                self.clf[i].apply(xaviern_init)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            self.clf1.apply(xavieru_init)
            self.clf2.apply(xavieru_init)
        elif act_func == "kn":
            self.clf1.apply(knorm_init)
            self.clf2.apply(knorm_init)
        elif act_func == "ku":          
            self.clf1.apply(kuni_init)  
            self.clf2.apply(kuni_init)  
        
    def forward(self, x, adj):
        out = []
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)

        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out    
    
class GCN_E2_decline_L2_div10_clf1_selu_multi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__()
        in_dim2 = int(in_dim/10)

        if act_func == "xn":
            act_func = "selu_xn"
        elif act_func == "xu":
            act_func = "selu_xu"
        self.out_dim = out_dim    
        
        self.gc1 = GraphConvolution(in_dim, in_dim2, act_func, dropout) 
        self.gc2 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(in_dim2, num_class)))
                            
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            for i in range(len(out_dim)):
                self.clf[i].apply(xaviern_init)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            self.clf1.apply(xavieru_init)
            self.clf2.apply(xavieru_init)
        elif act_func == "kn":
            self.clf1.apply(knorm_init)
            self.clf2.apply(knorm_init)
        elif act_func == "ku":          
            self.clf1.apply(kuni_init)  
            self.clf2.apply(kuni_init)  
        
    def forward(self, x, adj):
        out = []
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)

        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out   
    
class GCN_E2_decline_L2_log_clf1_selu_multi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__()
        in_dim2 = int(math.log(in_dim))

        if act_func == "xn":
            act_func = "selu_xn"
        elif act_func == "xu":
            act_func = "selu_xu"
        self.out_dim = out_dim    
        
        self.gc1 = GraphConvolution(in_dim, in_dim2, act_func, dropout) 
        self.gc2 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(in_dim2, num_class)))
                            
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            for i in range(len(out_dim)):
                self.clf[i].apply(xaviern_init)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            self.clf1.apply(xavieru_init)
            self.clf2.apply(xavieru_init)
        elif act_func == "kn":
            self.clf1.apply(knorm_init)
            self.clf2.apply(knorm_init)
        elif act_func == "ku":          
            self.clf1.apply(kuni_init)  
            self.clf2.apply(kuni_init)  
        
    def forward(self, x, adj):
        out = []
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)

        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out   
    
         
class GCN_E2_decline_L4_sqr_clf1_selu_multi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__()
        in_dim2 = int(math.sqrt(in_dim))

        if act_func == "xn":
            act_func = "selu_xn"
        elif act_func == "xu":
            act_func = "selu_xu"
        self.out_dim = out_dim    
        
        self.gc1 = GraphConvolution(in_dim, in_dim2, act_func, dropout) 
        self.gc2 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.gc3 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.gc4 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(in_dim2, num_class)))
                            
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            for i in range(len(out_dim)):
                self.clf[i].apply(xaviern_init)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            self.clf1.apply(xavieru_init)
            self.clf2.apply(xavieru_init)
        elif act_func == "kn":
            self.clf1.apply(knorm_init)
            self.clf2.apply(knorm_init)
        elif act_func == "ku":          
            self.clf1.apply(kuni_init)  
            self.clf2.apply(kuni_init)  
        
    def forward(self, x, adj):
        out = []
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        x = self.gc4(x, adj)

        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out   
    
class GCN_E2_decline_L5_sqr_clf1_selu_multi(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, act_func):
        super().__init__()
        in_dim2 = int(math.sqrt(in_dim))

        if act_func == "xn":
            act_func = "selu_xn"
        elif act_func == "xu":
            act_func = "selu_xu"
        self.out_dim = out_dim    
        
        self.gc1 = GraphConvolution(in_dim, in_dim2, act_func, dropout) 
        self.gc2 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.gc3 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.gc4 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.gc5 = GraphConvolution(in_dim2, in_dim2, act_func, dropout)
        self.clf = []
        self.clf = torch.nn.ModuleList(self.clf)
        for num_class in out_dim:
            self.clf.append(nn.Sequential(nn.Linear(in_dim2, num_class)))
                            
        if (act_func == "leaky_relu_xn") | (act_func == "selu_xn"):
            for i in range(len(out_dim)):
                self.clf[i].apply(xaviern_init)
        elif (act_func == "leaky_relu_xu") | (act_func == "selu_xu"):
            self.clf1.apply(xavieru_init)
            self.clf2.apply(xavieru_init)
        elif act_func == "kn":
            self.clf1.apply(knorm_init)
            self.clf2.apply(knorm_init)
        elif act_func == "ku":          
            self.clf1.apply(kuni_init)  
            self.clf2.apply(kuni_init)  
        
    def forward(self, x, adj):
        out = []
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x = self.gc3(x, adj)
        x = self.gc4(x, adj)
        x = self.gc5(x, adj)
        for i in range(len(self.out_dim)):
            out.append(self.clf[i](x))
        return out    

