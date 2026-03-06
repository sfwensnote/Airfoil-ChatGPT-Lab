# AGENTS.md - Coding Guidelines for Airfoil Lab

## Project Overview
Airfoil Lab is an AI-enhanced aerodynamic design platform with:
- **Frontend**: Next.js 16 + React 19 + TypeScript + Tailwind CSS v4
- **Backend**: FastAPI + Python 3.9+ + SQLite
- **AI System**: LangGraph multi-agent with local MLX inference

## Build & Development Commands

### Frontend (airfoil-lab-react/)
```bash
cd airfoil-lab-react
npm run dev          # Start dev server (localhost:3000)
npm run build        # Production build
npm run lint         # Run ESLint (Next.js config)
```

### Backend
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r agents/requirements.txt  # For AI features

# Start servers
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
./start.sh           # Starts MLX + Backend + Frontend
```

### Testing
**Note**: No test framework is currently configured. When adding tests:
- Frontend: Use Vitest or Jest for unit tests
- Backend: Use pytest for Python tests
- Run single test: `pytest test_file.py::test_function -v`

## Code Style Guidelines

### TypeScript/React (Frontend)

#### Imports Order
1. React/Next.js imports
2. Third-party libraries (lucide-react, zustand, etc.)
3. Local UI components (@/components/ui/*)
4. Feature components (@/components/*)
5. Stores (@/stores)
6. Utilities (@/lib/*, @/types)

#### Component Pattern
```typescript
'use client';  // For client components

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface Props {
  value: number;
  onChange: (val: number) => void;
}

export function ComponentName({ value, onChange }: Props) {
  const [local, setLocal] = useState(value);
  
  return (
    <div className="flex gap-2">
      <Button onClick={() => onChange(local)}>Action</Button>
    </div>
  );
}
```

#### Naming Conventions
- Components: PascalCase (e.g., `AirfoilPreview.tsx`)
- Hooks: camelCase with `use` prefix (e.g., `useSimulationStore`)
- Stores: camelCase with `use` prefix + `Store` suffix
- Types/Interfaces: PascalCase (e.g., `GeometryParams`)
- Utility files: camelCase (e.g., `geometry.ts`)

#### Styling (Tailwind v4)
- Use `cn()` utility for conditional classes
- Use Tailwind's arbitrary values sparingly: `w-[100px]`
- Prefer semantic color tokens: `bg-primary`, `text-destructive`
- Group related classes: layout → spacing → colors → effects

#### Error Handling
- Use Zustand store for global error state
- Prefer early returns for error cases
- Use toast notifications via `sonner` for user feedback

### Python (Backend)

#### File Structure
```python
# -*- coding: utf-8 -*-
"""
Module docstring with description.
"""

# Standard library imports
import os
from datetime import datetime

# Third-party imports
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Local imports
from agents.config import get_agent_config
```

#### Naming Conventions
- Functions/variables: snake_case (e.g., `run_xfoil_polar`)
- Classes: PascalCase (e.g., `AirfoilHistory`)
- Constants: UPPER_SNAKE_CASE (e.g., `DB_PATH`)
- Type aliases: PascalCase with descriptive names

#### Type Hints
- Use Python 3.9+ type hints: `dict`, `list`, `|` for unions
- Use Pydantic models for request/response validation
- Add return types to all functions

#### Error Handling
```python
# Prefer explicit error handling
try:
    result = compute_value(data)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    return {"status": "error", "message": str(e)}
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal error")
```

#### Database (SQLAlchemy)
- Use declarative base for models
- Add type hints with Mapped[] for columns
- Use context managers for sessions:
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Architecture Patterns

### State Management
- **Frontend**: Zustand stores in `src/stores/`
  - `useSimulationStore` - Geometry & environment
  - `useChatStore` - Chat messages & streaming
  - `useUserStore` - Auth & user history

### API Communication
- Use axios for HTTP requests
- Base URL configured via environment
- Handle errors with try/catch + toast notifications

### Multi-Agent System
- Agents defined in `agents/config.py`
- LangGraph workflow in `agents/graph.py`
- Each agent has: system prompt, temperature, model path
- Use `create_llm()` helper for consistent LLM configuration

## Git Workflow
1. Create feature branch: `git checkout -b feature/name`
2. Make focused commits with clear messages
3. Ensure lint passes before committing
4. Do NOT commit secrets or API keys

## Performance Guidelines
- Use React.memo for expensive components
- Debounce rapid state updates (sliders)
- Use useMemo for expensive calculations
- Lazy load heavy components with dynamic imports
