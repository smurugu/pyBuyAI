SET INTERPRETER=C:\Anaconda34_py36x64
SET PATH=%PATH%;%INTERPRETER%

SET WORKING_DIRECTORY=%~dp0

CD %WORKING_DIRECTORY%

ECHO "GRID SEARCH START"

CALL ./config_files\run_236d7736-eacc-46da-94cc-54a8ad5e09f9.bat
CALL ./config_files\run_1f5ec170-2f0d-48ed-a3d8-9552e0991b89.bat
CALL ./config_files\run_0af2d2b3-2fef-47a0-9d11-25b0add53c40.bat
CALL ./config_files\run_c00a2d8d-433e-4bf3-968d-79ace6639a73.bat
CALL ./config_files\run_b32157c2-9b1d-4711-915a-d9cca0c0cb33.bat
CALL ./config_files\run_64814084-38a3-494d-a280-c2fb7a89fb9f.bat

ECHO "GRID SEARCH FINISHED"