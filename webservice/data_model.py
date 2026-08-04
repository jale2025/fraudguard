"""Pydantic request and response models for the prediction API."""

from pydantic import BaseModel, Field


# This model describes the JSON body accepted by POST /predict.
class TransactionUnknownLabel(BaseModel):
    """
    Pydantic model describing a single transaction's features used for prediction with an unknown label.

    Attributes correspond to the anonymized V1..V28 features, the transaction Time,
    and Amount fields expected by the prediction API.
    """

    elapsed_sec: float = Field(..., examples=[2.0])
    pc_1: float = Field(..., examples=[-1.5])
    pc_2: float = Field(..., examples=[1.3])
    pc_3: float = Field(..., examples=[-1.7])
    pc_4: float = Field(..., examples=[0.1])
    pc_5: float = Field(..., examples=[-2.1])
    pc_6: float = Field(..., examples=[-3.8])
    pc_7: float = Field(..., examples=[1.5])
    pc_8: float = Field(..., examples=[-1.4])
    pc_9: float = Field(..., examples=[-0.5])
    pc_10: float = Field(..., examples=[0.3])
    pc_11: float = Field(..., examples=[-0.1])
    pc_12: float = Field(..., examples=[1.1])
    pc_13: float = Field(..., examples=[-1.0])
    pc_14: float = Field(..., examples=[1.5])
    pc_15: float = Field(..., examples=[1.4])
    pc_16: float = Field(..., examples=[-1.2])
    pc_17: float = Field(..., examples=[1.6])
    pc_18: float = Field(..., examples=[-1.9])
    pc_19: float = Field(..., examples=[-1.4])
    pc_20: float = Field(..., examples=[1.0])
    pc_21: float = Field(..., examples=[-1.7])
    pc_22: float = Field(..., examples=[-1.5])
    pc_23: float = Field(..., examples=[1.5])
    pc_24: float = Field(..., examples=[1.2])
    pc_25: float = Field(..., examples=[1.3])
    pc_26: float = Field(..., examples=[1.1])
    pc_27: float = Field(..., examples=[-1.2])
    pc_28: float = Field(..., examples=[1.5])
    amount: float = Field(..., examples=[255.65])


class TransactionKnownLabel(TransactionUnknownLabel):
    """
    Pydantic model describing a single transaction's features used for prediction with a known label.

    Attributes correspond to the anonymized V1..V28 features, the transaction Time,
    and Amount fields expected by the prediction API.
    """

    # The targte class represents the 'class' column
    # Was not possible to use 'class' - it's a reserved key word for classes
    target_class: int = Field(..., alias="class", examples=[0])

    # Allows the acces via t.target_class or field name
    model_config = {"populate_by_name": True}


# The response reuses every request feature and appends the model output.
class TransactionClassificationKnownLabel(TransactionKnownLabel):
    """
    Response model including the predicted class for a transaction with an unknown label.

    Inherits all transaction features from Transaction and appends a
    `prediction` field containing the model's integer class output.
    """

    prediction: int


class TransactionClassificationUnknownLabel(TransactionUnknownLabel):
    """
    Response model including the predicted class for a transaction with a known label.

    Inherits all transaction features from Transaction and appends a
    `prediction` field containing the model's integer class output.
    """

    prediction: int
