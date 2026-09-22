from typing import Optional
from pydantic import BaseModel, model_validator
from fastapi import Form


class SearchRequestDTO(BaseModel):
    brand: Optional[str] = None
    model_name: Optional[str] = None
    production_year: Optional[int] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    fuel_type: Optional[str] = None
    gearbox: Optional[str] = None


    @classmethod
    def as_form(
        cls,
        brand: Optional[str] = Form(None),
        model_name: Optional[str] = Form(None),
        production_year: Optional[int] = Form(None),
        price: Optional[float] = Form(None),
        currency: Optional[str] = Form(None),
        fuel_type: Optional[str] = Form(None),
        gearbox: Optional[str] = Form(None)
    ) -> "SearchRequestDTO":
        return cls(
            brand=brand,
            model_name=model_name,
            production_year=production_year,
            price=price,
            currency=currency,
            fuel_type=fuel_type,
            gearbox=gearbox
        )

    @model_validator(mode="after")
    def check_dependent_fields(self):
        if self.price is not None and self.currency is None:
            raise ValueError("enter currency")

        return self


class CarSearchResultDTO(BaseModel):
    url: str
    score: float
    brand: str
    model_name: str
    production_year: int
    price: float
    currency: str
    color: str
    mileage: int
    fuel_type: str
    gearbox: str
    body_type: str
    extra_info: str
    engine_capacity: Optional[int] = None
    engine_power: Optional[int] = None


class SearchResponseDTO(BaseModel):
    llm_response: str
    neighbors: list[CarSearchResultDTO]