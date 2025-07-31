@echo off

REM Get the current working directory
set "current_dir=%cd%"
echo %current_dir%

REM Activate the neutorch environment
echo Activating the neutorch environment...
call conda activate neutorch || (
    echo ERROR: Failed to activate neutorch environment.
    exit /b 1
)

REM Check if the environment was activated successfully
if %errorlevel% neq 0 (
    echo Failed to activate conda environment 'neutorch'. Please check if it exists.
    exit /b 1
)

REM Wait for the environment activation to stabilize
timeout /t 1 /nobreak > nul

echo Checking for Streamlit processes...

REM Run the WMIC command and process the output directly
for /f "skip=1 tokens=*" %%a in ('wmic process where "name='streamlit.exe'" get CommandLine^,ProcessId') do (
    REM Extract ProcessId (assuming it's the second token in the line)
    for /f "tokens=4" %%b in ("%%a") do (
        REM Display ProcessId
        echo ProcessId: %%b

        REM Kill the process using the extracted PID
        taskkill /F /PID %%b
        echo Killed ProcessId: %%b
    )
)

REM Now start the new Streamlit app in the background
echo Starting new Streamlit app...
streamlit run "%current_dir%\neuTorch_main.py"
