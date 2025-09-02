"""
Setup script for Azure ML credentials - Student version
Run this script once to configure your Azure CLI for the shared workspace
"""

import json
import os
import subprocess
import sys

# Your workspace credentials
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")


def run_command(cmd, description):
    """
    Run a shell command and handle errors with user-friendly messages.

    Parameters
    ----------
    cmd : str
        The shell command to execute.
    description : str
        A short description of the command/process being run, used for logging.

    Returns
    -------
    str or None
        The standard output from the command if successful; otherwise, None.
    """
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None


def main():
    """
    Set up and verify Azure ML Student Workspace environment.

    Checks Azure CLI installation, Azure ML extension, logs in user,
    sets subscription, verifies workspace and datastore access.

    Raises
    ------
    SystemExit
        If Azure CLI is not installed.

    Notes
    -----
    The function prints progress and error messages directly to stdout.
    User input is required for Azure login.
    """
    print("🎓 Azure ML Student Workspace Setup")
    print("=" * 50)

    # Check if Azure CLI is installed
    print("\n1. Checking Azure CLI installation...")
    result = run_command("az --version", "Azure CLI version check")
    if result is None:
        print("❌ Azure CLI not found. Please install it first:")
        print("   Download from: https://aka.ms/installazurecliwidows")
        sys.exit(1)

    # Check ML extension
    print("\n2. Checking Azure ML extension...")
    result = run_command("az extension list", "Extension list")
    if result and "ml" not in result:
        print("🔧 Installing Azure ML extension...")
        run_command("az extension add -n ml", "ML extension installation")
    else:
        print("✅ Azure ML extension already installed")

    # Login (this will open browser)
    print("\n3. Azure login...")
    print("⚠️  This will open your browser for authentication")
    input("Press Enter to continue with login...")
    run_command("az login", "Azure login")

    # Set subscription
    print(f"\n4. Setting subscription to {SUBSCRIPTION_ID}...")
    run_command(f"az account set --subscription {SUBSCRIPTION_ID}", "Subscription setting")

    # Verify workspace access
    print(f"\n5. Verifying workspace access...")
    cmd = (
        f"az ml workspace show --name {WORKSPACE_NAME} "
        f"--resource-group {RESOURCE_GROUP} --subscription {SUBSCRIPTION_ID}"
    )
    result = run_command(cmd, "Workspace verification")

    if result:
        try:
            workspace_info = json.loads(result)
            print(f"✅ Connected to workspace: {workspace_info.get('name', 'Unknown')}")
            print(f"   Location: {workspace_info.get('location', 'Unknown')}")
        except json.JSONDecodeError:
            print("⚠️  Workspace accessible but couldn't parse details")

    # Test datastore access
    print(f"\n6. Testing datastore access...")
    cmd = (
        f"az ml datastore show --name workspaceblobstore "
        f"--workspace-name {WORKSPACE_NAME} --resource-group {RESOURCE_GROUP} "
        f"--subscription {SUBSCRIPTION_ID}"
    )
    result = run_command(cmd, "Datastore verification")

    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nYour Azure CLI is now configured for:")
    print(f"   Subscription: {SUBSCRIPTION_ID}")
    print(f"   Resource Group: {RESOURCE_GROUP}")
    print(f"   Workspace: {WORKSPACE_NAME}")
    print("\nYou can now run your Streamlit app with Azure ML integration.")


if __name__ == "__main__":
    main()
