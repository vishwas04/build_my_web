import React, { Component } from "react";
import './App.css';
import {BrowserRouter as Router,Route,Link,Switch,Redirect} from "react-router-dom";
import Home from "./home.js"
import Edit from "./edit.js"

class App extends Component
{
    
    render()
    {
        
        return(
               
    <Switch>
    <Route exact path="/" component={Home} />
    <Route exact path="/edit" component={Edit} />
    </Switch>
               
               );
    }
}

export default App;
