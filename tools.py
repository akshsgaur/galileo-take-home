"""
Web search tools for the research agent.
Uses DuckDuckGo HTML scraping (no API key required).
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import time
import urllib.parse


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using DuckDuckGo HTML interface.

    Args:
        query: Search query string
        num_results: Maximum number of results to return (default: 5)

    Returns:
        List of dictionaries with 'title', 'url', and 'snippet' keys
    """
    # URL encode the query
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        # Find all result divs
        result_divs = soup.find_all('div', class_='result')

        for result_div in result_divs[:num_results]:
            try:
                # Extract title and URL
                title_elem = result_div.find('a', class_='result__a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)
                url = title_elem.get('href', '')

                # Extract snippet
                snippet_elem = result_div.find('a', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                # Only add if we have at least title and URL
                if title and url:
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
            except Exception as e:
                print(f"Error parsing result: {e}")
                continue

        return results

    except requests.exceptions.RequestException as e:
        print(f"Search request error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected search error: {e}")
        return []


if __name__ == "__main__":
    # Test the search function
    print("Testing DuckDuckGo search...")
    print("-" * 80)

    test_query = "AI observability trends 2024"
    print(f"Query: {test_query}\n")

    results = search_web(test_query, num_results=3)

    if results:
        print(f"Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
            print()
    else:
        print("No results found or search failed.")

    print("-" * 80)
