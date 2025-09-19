import os
from dotenv import load_dotenv
import sys

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from agent import ConservativeOptimizedCreditAgent
from base import logger
import logging

def main():
    """
    Main function to run the credit analysis agent.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Check if the API key is set
    if not os.getenv("DEEPSEEK_API_KEY"):
        logger.error("DEEPSEEK_API_KEY not found in .env file. Please add it to run the analysis.")
        return

    # Create agent configuration
    # In a real application, this would be loaded from a file
    config = {
        "data_collection": {
            "api_keys": {
                "qichacha": os.getenv("QICHACHA_API_KEY"),
                "tianyancha": os.getenv("TIANYANCHA_API_KEY")
            }
        }
    }

    # Instantiate and initialize the agent
    agent = ConservativeOptimizedCreditAgent(config)
    if not agent.initialize():
        logger.error("Agent initialization failed. Exiting.")
        return

    # Define company information for analysis
    company_info = {
        "company_name": "小米通讯技术有限公司",
        "industry": "信息技术"
    }

    # Run the credit analysis
    try:
        credit_data, risk_result = agent.analyze_credit(company_info)

        # Print the results
        logger.info("Credit Analysis Report:")
        logger.info(f"  Company Name: {credit_data.raw_data.get('enterprise_info', {}).get('company_name')}")
        logger.info(f"  Credit Score: {credit_data.credit_score}")
        logger.info(f"  Credit Rating: {credit_data.credit_rating}")
        logger.info(f"  Risk Level: {risk_result.risk_level}")
        logger.info("  Risk Details:")
        for factor, details in risk_result.risk_details.items():
            logger.info(f"    {factor}: {details}")

    except Exception as e:
        logger.error(f"An error occurred during credit analysis: {e}", exc_info=True)

if __name__ == "__main__":
    main()