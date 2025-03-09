import React, { useState } from 'react';
import { FaEnvelope, FaLinkedin } from 'react-icons/fa';
import { Link } from 'react-router-dom';

function Home() {
  const [showEmail, setShowEmail] = useState(false);

  const toggleEmail = () => {
    setShowEmail(!showEmail);
  };

  return (
    <div className="home">
      <h1>Bruce Fan</h1>
      <div className="bio-container">
        <div className="profile-pic-container">
          <img src="/bruce.png" alt="Bruce Fan" className="profile-pic" />
        </div>
        <div className="bio">
          <p>
            Passionate and aspiring computer science major at the University of Texas at Austin,
            with a strong foundation in application development and quantitative reasoning.
            Demonstrated history of academic excellence and innovative project development.
          </p> 
        </div>
      </div>
      <div className="contact-info">
        <div className="contact-link" onClick={toggleEmail}>
          <FaEnvelope className="contact-icon" />
          <span className="contact-text">
            {showEmail ? (
              <>
                bruce<span style={{display: 'none'}}>foo</span>xfan
                <span style={{display: 'none'}}>bar</span>@
                <span style={{display: 'none'}}>null</span>
                gmail<span style={{display: 'none'}}>foo</span>.com
              </>
            ) : (
              'Email'
            )}
          </span>
        </div>
        {}
      </div>
    </div>
  );
}

export default Home;
