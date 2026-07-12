"""ResLSTM với Multi-Vector Attention (v2 — reviewed & tối ưu cho Wav2Vec2 + RAVDESS).

Lớp `ResLSTM_Multi_Att` — tên class, tên hàm và toàn bộ chữ ký public API
(__init__, forward, get_feature_vector) được GIỮ NGUYÊN so với bản gốc để
không phá vỡ train.ipynb. Chỉ phần bên trong (internal) được sửa.

Tóm tắt các thay đổi so với v1 (chi tiết lý do xem trong phần trả lời kèm theo):
  1. Projection layer: thêm GELU + LayerNorm sau Linear(768->256) để projection
     có tính phi tuyến thay vì chỉ là một phép biến đổi affine.
  2. Multi-Vector Attention: thêm tanh() giữa attention_vector và head_combine.
     Đây là lỗi kiến trúc quan trọng nhất được phát hiện: composition của hai
     phép chiếu tuyến tính liên tiếp (matmul rồi Linear không bias) vẫn là một
     phép biến đổi tuyến tính duy nhất, khiến num_att > 1 KHÔNG có thêm sức biểu
     diễn nào so với num_att = 1. Thêm tanh() khắc phục vấn đề này.
  3. Embedding head: đổi BatchNorm1d -> LayerNorm cho feature_vector cuối cùng
     (self.bn), vì embedding này được dùng trực tiếp cho Triplet Loss với
     MPerClassSampler (mỗi batch có nhiều mẫu cùng lớp) — BatchNorm trong tình
     huống này rò rỉ thống kê batch giữa các mẫu cùng lớp (anchor/positive),
     làm bài toán triplet "dễ" hơn giả tạo trên tập train nhưng không tổng quát
     hoá tốt khi val/test có phân bố batch khác. LayerNorm không phụ thuộc vào
     các mẫu khác trong batch nên tránh được rò rỉ này.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MODEL_VERSION = "v2"


class ResLSTM_Multi_Att(nn.Module):
    """ResLSTM với residual giữa LSTM1 và LSTM2 + Variant A multi-vector attention."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_att: int,
                 num_classes: int,
                 projection_dim: int = 256,
                 projection_dropout: float = 0.3,
                 dropout_p: float = 0.1,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.input_size = input_size          # dim đặc trưng thô (768 với Wav2Vec2-base)
        self.projection_dim = projection_dim  # dim sau Projection == input/hidden của LSTM1
        self.num_layers = num_layers

        # --- Projection Layer: 768 -> 256 ---
        # LayerNorm đầu vào ổn định scale của hidden states Wav2Vec2 (không có
        # global mean/std normalization như pipeline MFCC cũ).
        self.layer_norm = nn.LayerNorm(input_size)
        self.projection = nn.Linear(input_size, projection_dim)
        # ĐÃ SỬA: thêm GELU + LayerNorm sau projection.
        # Lý do KHÔNG mở rộng qua một tầng ẩn 512 (768->512->256) như một số gợi ý
        # thường thấy: RAVDESS chỉ có ~1440 utterance, thêm một Linear(768,512) +
        # Linear(512,256) sẽ tăng đáng kể số tham số (~500K) trong khi lợi ích biểu
        # diễn cho bài toán 8 lớp là không cần thiết -> tăng nguy cơ overfitting mà
        # không có bằng chứng đánh đổi xứng đáng. Thay vào đó chỉ thêm phi tuyến +
        # chuẩn hoá ngay tại 256-dim, chi phí tham số gần như bằng 0 (chỉ 2*256 cho
        # LayerNorm) nhưng giúp projection không còn là một phép affine thuần tuý.
        self.projection_act = nn.GELU()
        self.projection_norm = nn.LayerNorm(projection_dim)
        self.projection_dropout = nn.Dropout(p=projection_dropout)

        self.lstm1 = nn.LSTM(projection_dim, projection_dim, num_layers,
                             bidirectional=False, batch_first=True)
        self.lstm2 = nn.LSTM(projection_dim, hidden_size, num_layers,
                             bidirectional=False, batch_first=True)

        self.attention_vector = nn.Parameter(torch.empty(hidden_size, num_att))
        self.head_combine = nn.Linear(num_att, 1, bias=False)

        self.bn_residual = nn.BatchNorm1d(projection_dim)
        # ĐÃ SỬA: BatchNorm1d -> LayerNorm cho embedding cuối cùng (xem docstring module).
        # Tên attribute `self.bn` được GIỮ NGUYÊN để không phá vỡ code bên ngoài
        # (dù không có nơi nào trong train.ipynb truy cập trực tiếp attribute này).
        self.bn = nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(hidden_size, num_classes, device=self.device)
        self.drop = nn.Dropout(p=dropout_p)

        self.classes = num_classes
        self.num_att = num_att

        self.initialize_model_weights()

    def initialize_model_weights(self):
        for lstm, proj_size in [(self.lstm1, self.projection_dim),
                                (self.lstm2, self.hidden_size)]:
            for layer in range(self.num_layers):
                nn.init.xavier_normal_(getattr(lstm, f'weight_ih_l{layer}'))
                nn.init.orthogonal_(getattr(lstm, f'weight_hh_l{layer}'))
                bias_ih = getattr(lstm, f'bias_ih_l{layer}')
                bias_hh = getattr(lstm, f'bias_hh_l{layer}')
                nn.init.zeros_(bias_ih)
                nn.init.zeros_(bias_hh)
                with torch.no_grad():
                    bias_ih[proj_size: 2 * proj_size].fill_(1.0)
                    bias_hh[proj_size: 2 * proj_size].fill_(1.0)
        nn.init.xavier_normal_(self.attention_vector)
        nn.init.xavier_uniform_(self.head_combine.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def get_feature_vector(self, lstm_out):
        attention_scores_per_head = torch.matmul(lstm_out, self.attention_vector)
        # ĐÃ SỬA: thêm tanh() phi tuyến giữa hai phép chiếu tuyến tính.
        # Nếu không có phi tuyến ở đây, matmul(x, attention_vector) rồi head_combine
        # (Linear không bias) tương đương một phép chiếu tuyến tính DUY NHẤT
        # x @ (attention_vector @ head_combine.weight.T) — tức là num_att "đầu"
        # attention chỉ là một cách tham số hoá lại (over-parameterize) của một
        # attention vector đơn, không có thêm khả năng biểu diễn nào (không có
        # nhiều "góc nhìn" khác nhau như ý tưởng multi-vector attention hướng tới).
        # tanh() phá vỡ tính tuyến tính này, làm mỗi head thực sự học một cách
        # tính điểm khác nhau trước khi được kết hợp.
        attention_scores_per_head = torch.tanh(attention_scores_per_head)
        attention_scores = self.head_combine(attention_scores_per_head).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        feature_vector = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return feature_vector, attention_weights

    def forward(self, x, lengths, return_embeddings=False, return_attention=False):
        batch_size = x.size(0)

        # --- Projection Layer: 768 -> 256 ---
        # x đầu vào là hidden states thô của Wav2Vec2 (B, T, 768).
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.projection_act(x)      # ĐÃ SỬA: GELU
        x = self.projection_norm(x)     # ĐÃ SỬA: LayerNorm(256) sau projection
        x = self.projection_dropout(x)
        x_original = x.clone()

        h0 = torch.zeros(self.num_layers, batch_size, self.projection_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.projection_dim).to(self.device)
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                        enforce_sorted=False)
        lstm1_out_packed, _ = self.lstm1(x_packed, (h0, c0))
        lstm1_out, _ = pad_packed_sequence(lstm1_out_packed, batch_first=True)

        if lstm1_out.size(1) != x.size(1):
            seq_len = min(lstm1_out.size(1), x.size(1))
            lstm1_out = lstm1_out[:, :seq_len, :]
            x_original = x_original[:, :seq_len, :]

        # Residual: cộng trước, chuẩn hoá sau (Post-Norm) — GIỮ NGUYÊN vị trí như bản
        # gốc. Đây là lựa chọn hợp lý ở đây: (a) mạng chỉ có 2 tầng LSTM (không sâu
        # như Transformer nhiều lớp nên vấn đề gradient vanishing của Post-Norm không
        # đáng ngại), (b) LSTM2 cần nhận input đã được chuẩn hoá ổn định về scale,
        # nên BatchNorm phải nằm SAU phép cộng residual chứ không phải trước.
        # KHÔNG thêm activation sau residual+BN: input vào LSTM2 nên giữ nguyên miền
        # giá trị (có thể âm) mà LSTM kỳ vọng ở gate/cell state; ép qua ReLU/GELU ở
        # đây sẽ cắt bỏ thông tin phía âm một cách không cần thiết trước khi vào một
        # tầng vốn đã có phi tuyến nội tại (sigmoid/tanh gates).
        residual = lstm1_out + x_original
        residual_norm = self.bn_residual(residual.transpose(1, 2)).transpose(1, 2)

        h0_l2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0_l2 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        residual_packed = pack_padded_sequence(residual_norm, lengths.cpu(),
                                               batch_first=True, enforce_sorted=False)
        lstm2_out_packed, _ = self.lstm2(residual_packed, (h0_l2, c0_l2))
        lstm2_out, _ = pad_packed_sequence(lstm2_out_packed, batch_first=True)

        feature_vector, attention_weights = self.get_feature_vector(lstm2_out)

        # 1. Trích xuất đặc trưng và chuẩn hóa cho Triplet Loss
        # ĐÃ SỬA: self.bn giờ là LayerNorm (xem lý do ở __init__ / docstring module).
        # Thứ tự BN/LayerNorm -> L2 Normalize -> (Dropout chỉ áp cho nhánh classifier)
        # -> FC được GIỮ NGUYÊN vì đã hợp lý: chuẩn hoá phân phối trước khi ép lên
        # mặt cầu đơn vị (L2 normalize), sau đó mới dropout riêng cho nhánh FC (không
        # dropout lên chính embedding dùng cho Triplet Loss, vì embeddings trả về ở
        # return_embeddings=True là biến trước dropout — dropout tạo tensor mới,
        # không sửa in-place biến `embeddings` gốc).
        embeddings = self.bn(feature_vector)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 2. Phân loại
        logits = self.fc(self.drop(embeddings))

        # 3. Điều hướng linh hoạt đầu ra dựa vào Flags
        if return_attention:
            return logits, attention_weights

        if return_embeddings:
            return logits, embeddings

        return logits