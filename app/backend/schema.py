from pydantic import BaseModel, Field

# Request Schema
class TextInput(BaseModel):
    """Request model for emotion prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Input text for emotion prediction",
        example="I am very happy today!"
    )

# Response Schema
class PredictionResponse(BaseModel):
    """Response model for prediction."""
    success: bool
    input_text: str
    emotion: str
    confidence: float | None = None

# Health Check Schema
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str

# API Info Schema
class APIInfo(BaseModel):
    """API information."""
    project: str
    version: str
    framework: str
    author: str