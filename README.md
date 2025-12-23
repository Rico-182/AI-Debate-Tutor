# DebateLab

A web application for practicing debate with an AI opponent. Create structured debates with multiple rounds, submit turns via text or audio transcription, and get AI-generated responses.

## Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation and models
- **OpenAI API** - GPT-4o-mini for debate responses and Whisper-1 for audio transcription
- **python-dotenv** - Environment variable management

### Frontend
- **React** - UI library
- **Vite** - Build tool and dev server
- **Fetch API** - For HTTP requests

## Prerequisites

- Python 3.9 or higher
- Node.js 16+ and npm (or yarn)
- pip (Python package manager)
- OpenAI API key (optional, but required for full functionality)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd AI-Debate-Tutor
```

### 2. Install Python Dependencies

Install the required packages:

```bash
pip install fastapi uvicorn openai pydantic python-dotenv
```

Or if you prefer using a virtual environment (recommended):

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows
pip install fastapi uvicorn openai pydantic python-dotenv
```

## Configuration

### Environment Variables

Create a `.env` file in the `backend` directory:

```bash
cd backend
touch .env
```

Add your OpenAI API key to the `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

**Important:** 
- The variable name must be `OPENAI_API_KEY` (not `OPEN_API_KEY`)
- The `.env` file is already in `.gitignore` to keep your API key secure
- The app will work without an API key but will use stub responses for AI turns

## Running the Application

### 1. Start the Backend Server

Navigate to the `backend` directory and start the server:

```bash
cd backend
python3 -m uvicorn app.main:app --reload --port 8000
```

**Note:** Use `python3 -m uvicorn` instead of just `uvicorn` if the command is not found in your PATH.

The server will start at `http://localhost:8000`. You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### 2. Install Frontend Dependencies

Navigate to the `frontend` directory and install Node.js dependencies:

```bash
cd frontend
npm install
```

### 3. Start the Frontend Development Server

```bash
npm run dev
```

The React app will start at `http://localhost:3000` and should automatically open in your browser.

The frontend is configured to connect to `http://localhost:8000` by default. You can change the API base URL in the UI if needed.

## API Endpoints

- `GET /v1/health` - Health check
- `POST /v1/debates` - Create a new debate
- `GET /v1/debates/{debate_id}` - Get debate state and messages
- `POST /v1/debates/{debate_id}/turns` - Submit a turn
- `POST /v1/debates/{debate_id}/auto-turn` - Generate AI assistant turn
- `POST /v1/debates/{debate_id}/finish` - Finish debate early
- `POST /v1/transcribe` - Transcribe audio file

## Troubleshooting

### "command not found: uvicorn"

Use `python3 -m uvicorn` instead of just `uvicorn`:
```bash
python3 -m uvicorn app.main:app --reload --port 8000
```

### "Address already in use" (Port 8000)

Kill the process using port 8000:
```bash
lsof -ti:8000 | xargs kill
```

Or use a different port:
```bash
python3 -m uvicorn app.main:app --reload --port 8001
```
Then update the API base URL in the frontend UI.

### Python Version Compatibility

This project requires Python 3.9+. If you encounter syntax errors with `str | None`, ensure you're using Python 3.9 or higher. The code uses `Optional[str]` for compatibility.

### API Key Not Working

- Verify the `.env` file is in the `backend` directory
- Check that the variable name is exactly `OPENAI_API_KEY` (case-sensitive)
- Restart the server after creating or modifying the `.env` file
- Remove quotes if your API key is wrapped in quotes (some systems handle this differently)

## Development

### Frontend Development

The frontend uses Vite for fast development with hot module replacement. The app will automatically reload when you make changes.

To build for production:
```bash
cd frontend
npm run build
```

The built files will be in the `frontend/dist` directory.

### Backend Development

The backend uses in-memory storage, so data is not persisted between server restarts. This is suitable for development and testing.

For production deployment, consider:
- Adding a database (PostgreSQL, MongoDB, etc.)
- Implementing user authentication
- Adding rate limiting
- Setting up proper CORS configuration
- Using environment-specific configuration

## Deployment

This application can be deployed using Railway for the backend and Vercel for the frontend.

### Prerequisites

- GitHub account
- Railway account (sign up at [railway.app](https://railway.app))
- Vercel account (sign up at [vercel.com](https://vercel.com))
- OpenAI API key

### Backend Deployment (Railway)

1. **Push your code to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Create a new Railway project**:
   - Go to [railway.app](https://railway.app) and sign in
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will auto-detect the Python backend

3. **Configure environment variables**:
   - In your Railway project dashboard, go to "Variables"
   - Add the following environment variables:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     CORS_ORIGINS=https://your-frontend-url.vercel.app
     SCORING_MODEL=gpt-4o
     ```
   - **Important**: Replace `your-frontend-url.vercel.app` with your actual Vercel deployment URL

4. **Deploy**:
   - Railway will automatically detect the `Procfile` and start the server
   - The backend will be available at `https://your-project-name.up.railway.app`
   - Note the deployment URL - you'll need it for the frontend

5. **Verify deployment**:
   - Visit `https://your-backend-url.up.railway.app/v1/health` to verify the backend is running
   - Visit `https://your-backend-url.up.railway.app/docs` to view API documentation

### Frontend Deployment (Vercel)

1. **Install Vercel CLI** (optional, you can also use the web interface):
   ```bash
   npm i -g vercel
   ```

2. **Deploy from GitHub** (recommended):
   - Go to [vercel.com](https://vercel.com) and sign in
   - Click "Add New Project"
   - Import your GitHub repository
   - Configure the project:
     - **Root Directory**: `frontend`
     - **Framework Preset**: Vite
     - **Build Command**: `npm run build`
     - **Output Directory**: `dist`

3. **Set environment variables**:
   - In your Vercel project settings, go to "Environment Variables"
   - Add:
     ```
     VITE_API_BASE_URL=https://your-backend-url.up.railway.app
     ```
   - Replace `your-backend-url.up.railway.app` with your Railway backend URL

4. **Deploy**:
   - Click "Deploy"
   - Vercel will build and deploy your frontend
   - Your app will be available at `https://your-project-name.vercel.app`

5. **Update CORS in Railway**:
   - Go back to Railway dashboard
   - Update the `CORS_ORIGINS` variable to include your Vercel URL:
     ```
     CORS_ORIGINS=https://your-project-name.vercel.app
     ```
   - Railway will automatically redeploy with the new CORS settings

### Environment Variables Reference

#### Backend (Railway)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key |
| `CORS_ORIGINS` | No | `*` | Comma-separated list of allowed origins (e.g., `https://app.vercel.app`) |
| `SCORING_MODEL` | No | `gpt-4o` | Model to use for scoring (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`) |
| `SPEECH_CORPUS_DIR` | No | `./app/corpus` | Directory for RAG corpus files |
| `PORT` | Auto-set | - | Port number (automatically set by Railway) |

#### Frontend (Vercel)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_BASE_URL` | Yes | `http://localhost:8000` | Backend API URL (your Railway deployment URL) |

### Troubleshooting Deployment

#### Backend Issues

- **Port errors**: Railway automatically sets `PORT` - don't override it
- **Build failures**: Ensure `requirements.txt` includes all dependencies
- **CORS errors**: Verify `CORS_ORIGINS` includes your Vercel URL (no trailing slash)
- **API key errors**: Double-check `OPENAI_API_KEY` is set correctly in Railway

#### Frontend Issues

- **API connection errors**: Verify `VITE_API_BASE_URL` points to your Railway backend
- **Build failures**: Ensure all dependencies are in `package.json`
- **CORS errors**: Check that Railway `CORS_ORIGINS` includes your Vercel URL

### Example `.env` File (Backend)

Create a `.env` file in the `backend` directory for local development:

```bash
# Copy backend/.env.example to backend/.env and fill in your values
OPENAI_API_KEY=your_openai_api_key_here
CORS_ORIGINS=http://localhost:3000
SCORING_MODEL=gpt-4o
SPEECH_CORPUS_DIR=./app/corpus
```

**Note**: The `.env` file is gitignored and should never be committed to version control.

## License

[Add your license here]

