import React from 'react';
import { Container, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';

function ErrorPage() {
    let navigate = useNavigate();

    return (
        <Container className="d-flex flex-column justify-content-center align-items-center" style={{ minHeight: '80vh' }}>
            <h2>Something went wrong...</h2>
            <p>Please try a different search query or check back later.</p>
            <Button onClick={() => navigate('/')}>Go Back Home</Button>
        </Container>
    );
}

export default ErrorPage;
