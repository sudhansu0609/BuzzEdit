"""Free stock API keys (Pexels / Pixabay) — see store.app_settings and
presentation.stock. Kept out of routes/settings.py's generic GET/PUT `/`
because that route returns app_settings verbatim; keys must never come back
in full, only masked.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from store.app_settings import get_stock_api_keys_masked, set_stock_api_keys

router = APIRouter()


class StockKeysRequest(BaseModel):
    pexels_api_key: Optional[str] = None
    pixabay_api_key: Optional[str] = None


@router.get("/stock_keys")
async def get_stock_keys():
    """Masked state of the free-stock fallback keys. Never returns a key in
    full — only whether it is set and its last 4 characters."""
    return get_stock_api_keys_masked()


@router.put("/stock_keys")
async def put_stock_keys(body: StockKeysRequest):
    """Persist one or both keys. A field left out of the body leaves that key
    unchanged; sending "" clears it."""
    return set_stock_api_keys(pexels_api_key=body.pexels_api_key,
                              pixabay_api_key=body.pixabay_api_key)
