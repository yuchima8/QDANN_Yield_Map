""" basic models for yield prediction
encoder
predictor
"""
import torch
import torch.nn.functional as F
from torch import nn
from functions import ReverseLayerF

activation = nn.ReLU()

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, hidden_size)
        x = torch.tanh(self.linear1(inputs))
        # x shape: (batch_size, seq_len, hidden_size)
        x = self.linear2(x).squeeze(-1)
        # x shape: (batch_size, seq_len)
        x = torch.exp(x - x.max(dim=1, keepdim=True)[0])
        # x shape: (batch_size, seq_len)
        attn_weights = x / x.sum(dim=1, keepdim=True)
        # attn_weights shape: (batch_size, seq_len)
        attn_outputs = torch.bmm(attn_weights.unsqueeze(1), inputs).squeeze(1)
        # attn_outputs shape: (batch_size, hidden_size)
        return attn_outputs, attn_weights


class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, hidden_size):
        super(Multi_Head_Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(args.hidden_size, self.all_head_size)
        self.key = Linear(args.hidden_size, self.all_head_size)
        self.value = Linear(args.hidden_size, self.all_head_size)

        self.out = Linear(args.hidden_size, args.hidden_size)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # DEBUG(f'x.shape: {x.shape}')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # DEBUG(f'new_x_shape: {new_x_shape}')
        x = x.view(*new_x_shape)
        # DEBUG(f'x.shape: {x.shape}')
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # DEBUG(f'hidden_states.shape: {hidden_states.shape}')
        mixed_query_layer = self.query(hidden_states)
        # DEBUG(f'mixed_query_layer.shape: {mixed_query_layer.shape}')
        mixed_key_layer = self.key(hidden_states)
        # DEBUG(f'mixed_key_layer.shape: {mixed_key_layer.shape}')
        mixed_value_layer = self.value(hidden_states)
        # DEBUG(f'mixed_value_layer.shape: {mixed_value_layer.shape}')

        query_layer = self.transpose_for_scores(mixed_query_layer)
        # DEBUG(f'query_layer.shape: {query_layer.shape}')
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # DEBUG(f'key_layer.shape: {key_layer.shape}')
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # DEBUG(f'value_layer.shape: {value_layer.shape}')

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # DEBUG(f'attention_scores.shape: {attention_scores.shape}')
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # DEBUG(f'attention_scores.shape: {attention_scores.shape}')
        attention_probs = self.softmax(attention_scores)
        # DEBUG(f'attention_probs.shape: {attention_probs.shape}')
        weights = attention_probs
        # DEBUG(f'weights.shape: {weights.shape}')

        context_layer = torch.matmul(attention_probs, value_layer)
        # DEBUG(f'context_layer.shape: {context_layer.shape}')
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # DEBUG(f'context_layer.shape: {context_layer.shape}')
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # DEBUG(f'{context_layer.size()[:-2]} {(self.all_head_size,)} {self.all_head_size}')
        # DEBUG(f'new_context_layer_shape: {new_context_layer_shape}')
        context_layer = context_layer.view(*new_context_layer_shape)
        # DEBUG(f'context_layer.shape: {context_layer.shape}')
        attention_output = self.out(context_layer)
        # DEBUG(f'attention_output.shape: {attention_output.shape}')

        # pdb.set_trace()
        return attention_output, weights


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x: [batch, 1, input_dim]
        queries = self.query(x) # [batch, 1, input_dim]
        keys = self.key(x) # [batch, 1, input_dim]
        values = self.value(x) # [batch, 1, input_dim]
        scores =  torch.mul(queries, keys) / (self.input_dim ** 0.5) # [batch, 1, input_dim] * [batch, input_dim, 1] torch.bmm(queries, keys.transpose(1, 2))
        attention = self.softmax(scores) # 1
        weighted = torch.mul(attention, values) # 1 * [batch, 1, input_dim] optimally: [batch, 1, weight] * [batch, 1, input_dim]   bmm

        return weighted, attention

class Encoder(nn.Module):
    def __init__(self, num_predictors, num_shared_feature, dropout_ratio):

        super(Encoder, self).__init__()

        num_neuron_h1 = 64
        num_neuron_h2 = 64
        num_neuron_h3 = 64
        num_neuron_h4 = 64
        num_neuron_h5 = 64

        #self.encoder = nn.Sequential()

        self.dp = nn.Dropout(p=dropout_ratio)
        self.ac = activation

        #self.attn1 = SelfAttention(num_predictors)

        self.fc_1 = nn.Linear(num_predictors, num_neuron_h1)
        self.f_bn1 = nn.BatchNorm1d(num_neuron_h1)

        self.fc_2 = nn.Linear(num_neuron_h1, num_neuron_h2)
        self.f_bn2 = nn.BatchNorm1d(num_neuron_h2) #nn.Dropout(p=dropout_ratio)

        self.fc_3 = nn.Linear(num_neuron_h2, num_neuron_h3)
        self.f_bn3 = nn.BatchNorm1d(num_neuron_h3)  # nn.Dropout(p=dropout_ratio)

        self.fc_4 = nn.Linear(num_neuron_h3, num_neuron_h4)
        self.f_bn4 = nn.BatchNorm1d(num_neuron_h4)  # nn.Dropout(p=dropout_ratio)

        self.fc_5 = nn.Linear(num_neuron_h4, num_shared_feature)
        self.f_bn5 = nn.BatchNorm1d(num_shared_feature)  # nn.Dropout(p=dropout_ratio)

        #self.fc_6 = nn.Linear(num_neuron_h5, num_shared_feature)


    def _init_weights(self):

        nn.init.xavier_uniform_(self.fc_1.weight)
        nn.init.xavier_uniform_(self.fc_2.weight)
        nn.init.xavier_uniform_(self.fc_3.weight)

        nn.init.normal_(self.fc_1.bias, std=1e-6)
        nn.init.normal_(self.fc_2.bias, std=1e-6)
        nn.init.normal_(self.fc_3.bias, std=1e-6)

    def forward(self, x):

        #x, attn = self.attn1(x.unsqueeze(1))
        #x = x.squeeze(1)
        attn = 1

        x1 = self.fc_1(x)
        x1 = self.f_bn1(x1)
        x1 = self.dp(x1)
        x1 = self.ac(x1)

        x2 = self.fc_2(x1)
        x2 = self.f_bn2(x2)
        x2 = self.dp(x2)
        x2 = self.ac(x2)

        x3 = self.fc_3(x2)
        x3 = self.f_bn3(x3)
        x3 = self.dp(x3)
        x3 = self.ac(x3)

        x4 = self.fc_4(x3)
        x4 = self.f_bn4(x4)
        x4 = self.dp(x4)
        x4 = self.ac(x4)

        #Layer5
        output = self.fc_5(x4)

        return output, attn, x1, x2, x3, x4


class Predictor(nn.Module):
    def __init__(self, num_shared_feature, dropout_ratio):

        super(Predictor, self).__init__()

        # local single predictor
        num_neuron_c1 = 64
        num_neuron_c2 = 64

        #____
        self.dp = nn.Dropout(p=dropout_ratio)
        self.ac = activation

        self.r_bn0 = nn.BatchNorm1d(num_shared_feature)

        self.r_fc1 = nn.Linear(num_shared_feature, num_neuron_c1)
        self.r_bn1 = nn.BatchNorm1d(num_neuron_c1)

        self.r_fc2 = nn.Linear(num_neuron_c1, num_neuron_c2)
        self.r_bn2 = nn.BatchNorm1d(num_neuron_c2)

        self.r_fc3 = nn.Linear(num_neuron_c2, 1)

        #self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.r_fc1.weight)
        nn.init.xavier_uniform_(self.r_fc2.weight)
        nn.init.xavier_uniform_(self.r_fc3.weight)

        nn.init.normal_(self.r_fc1.bias, std=1e-6)
        nn.init.normal_(self.r_fc2.bias, std=1e-6)
        nn.init.normal_(self.r_fc3.bias, std=1e-6)

    def forward(self, x):

        x = self.r_bn0(x)

        x = self.r_fc1(x)
        x = self.r_bn1(x)
        x = self.dp(x)
        x = self.ac(x)

        x = self.r_fc2(x)
        x = self.r_bn2(x)
        x = self.dp(x)
        x = self.ac(x)

        output = self.r_fc3(x)

        return output


class Discriminator(nn.Module):

    def __init__(self, num_shared_feature, dropout_ratio):

        super(Discriminator, self).__init__()

        num_neuron_c1 = 64
        num_neuron_c2 = 64

        self.dp = nn.Dropout(p=dropout_ratio)
        self.ac = activation

        self.d_fc1 = nn.Linear(num_shared_feature, num_neuron_c1)
        self.d_bn1 = nn.BatchNorm1d(num_neuron_c1)

        self.d_fc2 = nn.Linear(num_neuron_c1, num_neuron_c2)
        self.d_bn2 = nn.BatchNorm1d(num_neuron_c2)

        self.d_fc3 = nn.Linear(num_neuron_c2, 2)

        #self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.d_fc1.weight)
        nn.init.xavier_uniform_(self.d_fc2.weight)
        nn.init.xavier_uniform_(self.d_fc3.weight)

        nn.init.normal_(self.d_fc1.bias, std=1e-6)
        nn.init.normal_(self.d_fc2.bias, std=1e-6)
        nn.init.normal_(self.d_fc3.bias, std=1e-6)

    def forward(self, feature, alpha):

        x = ReverseLayerF.apply(feature, alpha)
        #x = feature

        x = self.d_fc1(x)
        x = self.d_bn1(x)
        x = self.dp(x)
        x = self.ac(x)

        x = self.d_fc2(x)
        x = self.d_bn2(x)
        x = self.dp(x)
        x = self.ac(x)

        domain_output = self.d_fc3(x)

        return domain_output
