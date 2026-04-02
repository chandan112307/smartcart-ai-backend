"""Custom exception classes and error handlers."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class SmartCartException(Exception):
    """Base exception for SmartCart AI."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidQueryException(SmartCartException):
    def __init__(self, message: str = "Invalid or empty query"):
        super().__init__(message, status_code=400)


class AgentException(SmartCartException):
    def __init__(self, agent: str, message: str):
        super().__init__(f"Agent '{agent}' failed: {message}", status_code=500)


class LLMException(SmartCartException):
    def __init__(self, message: str = "LLM call failed"):
        super().__init__(message, status_code=502)


class DataLayerException(SmartCartException):
    def __init__(self, message: str = "Data retrieval failed"):
        super().__init__(message, status_code=503)


class CacheException(SmartCartException):
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message, status_code=500)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI application."""

    @app.exception_handler(SmartCartException)
    async def smartcart_exception_handler(
        request: Request, exc: SmartCartException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "query": "", "results": [], "best_option": {}, "deals": [], "total_price": 0},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "query": "", "results": [], "best_option": {}, "deals": [], "total_price": 0},
        )
