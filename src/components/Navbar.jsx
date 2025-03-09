import React from 'react';
import { NavLink } from 'react-router-dom';

function Navbar({ show }) {
  return (
    <nav className={`navbar ${show ? 'show' : 'hide'}`}>
      <div className="navbar-container">
        <NavLink to="/" className="navbar-logo">
          Adam Zweiger
        </NavLink>
        <ul className="nav-menu">
          <li className="nav-item">
            <NavLink to="/" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
              Home
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/projects" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
              Projects
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/blog" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
              Blog
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/resume" className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
              Resume
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
