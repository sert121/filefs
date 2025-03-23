import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizer
from typing import Dict, List, Optional, Tuple

class LegalBertForSequenceClassification(BertPreTrainedModel):
    def __init__(
        self,
        config,
        num_labels: int = 2,
        dropout: float = 0.1,
        use_legal_attention: bool = True
    ):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Legal domain-specific attention layer
        if use_legal_attention:
            self.legal_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=8,
                dropout=config.attention_probs_dropout_prob
            )
        else:
            self.legal_attention = None
            
        self.init_weights()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        legal_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        pooled_output = outputs[1]
        
        # Apply legal domain attention if enabled
        if self.legal_attention is not None:
            attention_output, _ = self.legal_attention(
                pooled_output.unsqueeze(0),
                pooled_output.unsqueeze(0),
                pooled_output.unsqueeze(0),
                key_padding_mask=legal_attention_mask
            )
            pooled_output = attention_output.squeeze(0)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class LegalBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = True,
        do_basic_tokenize: bool = True,
        never_split: Optional[List[str]] = None,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool = None,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case,
            do_basic_tokenize,
            never_split,
            unk_token,
            sep_token,
            pad_token,
            cls_token,
            mask_token,
            tokenize_chinese_chars,
            strip_accents,
            **kwargs
        )
        
        # Add legal domain-specific tokens
        self.add_special_tokens({
            "additional_special_tokens": [
                "[CASE]",
                "[STATUTE]",
                "[PRECEDENT]",
                "[ARGUMENT]",
                "[HOLDING]"
            ]
        })
        
    def encode_legal_text(
        self,
        text: str,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """Encode legal text with domain-specific processing."""
        # Split text into sections
        sections = self._split_legal_sections(text)
        
        # Process each section
        encoded_sections = []
        for section in sections:
            encoded = self(
                section,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors
            )
            encoded_sections.append(encoded)
            
        # Combine encoded sections
        combined = {
            "input_ids": torch.cat([e["input_ids"] for e in encoded_sections], dim=0),
            "attention_mask": torch.cat([e["attention_mask"] for e in encoded_sections], dim=0),
            "token_type_ids": torch.cat([e["token_type_ids"] for e in encoded_sections], dim=0)
        }
        
        return combined
        
    def _split_legal_sections(self, text: str) -> List[str]:
        """Split legal text into meaningful sections."""
        sections = []
        current_section = []
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                if current_section:
                    sections.append(" ".join(current_section))
                    current_section = []
            else:
                current_section.append(line)
                
        if current_section:
            sections.append(" ".join(current_section))
            
        return sections

class LegalBertTrainer:
    def __init__(
        self,
        model: LegalBertForSequenceClassification,
        tokenizer: LegalBertTokenizer,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def train_step(
        self,
        texts: List[str],
        labels: torch.Tensor,
        max_length: int = 512
    ) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Encode texts
        encoded = self.tokenizer.encode_legal_text(
            texts,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(**encoded, labels=labels)
        loss = outputs[0]
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}
        
    def evaluate(
        self,
        texts: List[str],
        labels: torch.Tensor,
        max_length: int = 512
    ) -> Dict[str, float]:
        """Evaluate model on given texts."""
        self.model.eval()
        
        with torch.no_grad():
            # Encode texts
            encoded = self.tokenizer.encode_legal_text(
                texts,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(**encoded, labels=labels)
            loss = outputs[0]
            logits = outputs[1]
            
            # Compute metrics
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == labels).float().mean()
            
            return {
                "loss": loss.item(),
                "accuracy": accuracy.item()
            } 