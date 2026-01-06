# -*- coding: utf-8 -*-
"""
Classe de base pour tous les indicateurs techniques.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Dict, Any

class BaseIndicator(ABC):
    """Classe de base pour tous les indicateurs techniques."""
    
    def __init__(self, name: str):
        self.name = name
        self._cache = {}
    
    @abstractmethod
    def calculate(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.Series, Dict[str, pd.Series]]:
        """Calcule l'indicateur technique."""
        pass
    
    def validate_data(self, data: pd.Series, min_length: int = 1) -> bool:
        """Valide les donn�es d'entr�e."""
        if not isinstance(data, pd.Series):
            return False
        
        if len(data) < min_length:
            return False
        
        if data.isna().all():
            return False
        
        return True
