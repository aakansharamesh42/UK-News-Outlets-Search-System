import axios from 'axios';

const BASE_URL = process.env.REACT_APP_ENDPOINT_URL;

export default BASE_URL;

export const fetchSearchResults = async (query, type, page = 1) => {
    // Remove empty parameters from the request
    const params = new URLSearchParams();
    params.append('q', query);
    if (page) params.append('page', page);

    const url = `${BASE_URL}/search/${type}?${params.toString()}`;

    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch search results');
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching search results:', error);
        throw error;
    }
};

// Function to call the query expansion endpoint
export const fetchQueryExpansions = async (query, expansions) => {
  try {
    //add the url of the query expansion endpoint in response
    const response = await axios.get(`${BASE_URL}/search/query-expansion?q=${query}&expansions=${expansions}`);
    return response.data;

  } catch (error) {
    console.error('Error fetching query expansions:', error);
    return [];
  }
};


export const postSearchTest = async (data) => {
    try {
        const response = await fetch(`${BASE_URL}/search/test`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        if (!response.ok) {
            throw new Error('Failed to post search test');
        }
        return await response.json();
    } catch (error) {
        console.error('Error posting search test:', error);
        throw error;
    }
};

export const fetchSearchBoolean = async (query, page = 1, limit = 10) => {
    try {
      const url = `${BASE_URL}/search/boolean?q=${encodeURIComponent(query)}&page=${page}&limit=${limit}`;
      const response = await fetch(url, {
        headers: {
          'Accept': 'application/json'
        }
      });
      if (!response.ok) {
        throw new Error(`Network response was not ok (status: ${response.status})`);
      }
      const data = await response.json();
      return data.results; // Assuming the API wraps the results in a "results" key
    } catch (error) {
      console.error('There was a problem fetching the boolean search results:', error);
      throw error;
    }
  };

export const fetchSearchTfidf = async (query, page = 1, limit = 10) => {
    try {
      const url = `${BASE_URL}/search/tfidf?q=${encodeURIComponent(query)}&page=${page}&limit=${limit}`;
      const response = await fetch(url, {
        headers: {
          'accept': 'application/json'
        }
      });
      if (!response.ok) {
        throw new Error(`Network response was not ok (status: ${response.status})`);
      }
      const data = await response.json();
      return data.results; // adjust this depending on the shape of your API response.
    } catch (error) {
      console.error('There was a problem fetching the TF-IDF search results:', error);
      throw error; // Re-throw the error so it can be caught and handled by the caller.
    }
};

export const fetchSpellChecker = async (query) => {
  try {
    const url = `${BASE_URL}/search/spellcheck?q=${encodeURIComponent(query)}`;
    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json'
      }
    });
    if (!response.ok) {
      throw new Error(`Network response was not ok (status: ${response.status})`);
    }
    const data = await response.text();
    return data; // Assuming the API wraps the results in a "results" key
  } catch (error) {
    console.error('There was a problem fetching the boolean search results:', error);
    throw error;
  }
};

