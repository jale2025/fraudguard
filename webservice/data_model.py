"""Pydantic request and response models for the prediction API."""

from pydantic import BaseModel, Field


# This model describes the JSON body accepted by POST /predict.
class Transaction(BaseModel):
    """
    Pydantic model describing a single transaction's features used for prediction.

    Attributes correspond to the anonymized V1..V28 features, the transaction Time,
    and Amount fields expected by the prediction API.
    """

    Time: float = Field(..., examples=[2.0])
    V1: float = Field(..., examples=[-1.5])
    V2: float = Field(..., examples=[1.3])
    V3: float = Field(..., examples=[-1.7])
    V4: float = Field(..., examples=[0.1])
    V5: float = Field(..., examples=[-2.1])
    V6: float = Field(..., examples=[-3.8])
    V7: float = Field(..., examples=[1.5])
    V8: float = Field(..., examples=[-1.4])
    V9: float = Field(..., examples=[-0.5])
    V10: float = Field(..., examples=[0.3])
    V11: float = Field(..., examples=[-0.1])
    V12: float = Field(..., examples=[1.1])
    V13: float = Field(..., examples=[-1.0])
    V14: float = Field(..., examples=[1.5])
    V15: float = Field(..., examples=[1.4])
    V16: float = Field(..., examples=[-1.2])
    V17: float = Field(..., examples=[1.6])
    V18: float = Field(..., examples=[-1.9])
    V19: float = Field(..., examples=[-1.4])
    V20: float = Field(..., examples=[1.0])
    V21: float = Field(..., examples=[-1.7])
    V22: float = Field(..., examples=[-1.5])
    V23: float = Field(..., examples=[1.5])
    V24: float = Field(..., examples=[1.2])
    V25: float = Field(..., examples=[1.3])
    V26: float = Field(..., examples=[1.1])
    V27: float = Field(..., examples=[-1.2])
    V28: float = Field(..., examples=[1.5])
    Amount: float = Field(..., examples=[255.65])

# The response reuses every request feature and appends the model output.
class TransactionClassification(Transaction):
    """
    Response model including the predicted class for a transaction.

    Inherits all transaction features from Transaction and appends a
    `prediction` field containing the model's integer class output.
    """

    prediction: int
