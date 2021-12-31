// import './home.css';
import React, { Component } from "react";
import axios from 'axios';
import './edit.css';
import { saveAs } from 'file-saver';
import  { useState } from 'react';
import fileDownload from 'js-file-download'
class Edit extends React.Component {

    constructor(props) {
        super(props);
        this.getData=this.getData.bind(this);
        this.call=this.call.bind(this);
        this.resiz=this.resiz.bind(this);
        this.slideshow=this.slideshow.bind(this);
        this.image=this.image.bind(this);
        this.sidebar=this.sidebar.bind(this);
        this.endbar=this.endbar.bind(this);
        this.navbar=this.navbar.bind(this);
        this.text=this.text.bind(this);
        this.ustate=this.ustate.bind(this);
        this.showAlert=this.showAlert.bind(this);
        this.handleDownload=this.handleDownload.bind(this);
        this.colorpicker=this.colorpicker.bind(this);
        this.state={
            data:{},
            first:1,
            no_of_slideshow:0,
            s_show:[],

            pres_ss:{},
            data_ss:{},
            data_text:0,
            data_img:{},


            nav_state:[],//home
            side_state:[],
            end_state:[],

            lnav_state:[],//link for home
            lside_state:[],
            lend_state:[]
            
        }
        
    }
    

    componentDidMount(){
        this.getData();
      }
    getData(){

    const url = 'http://localhost:5000/edit';
    fetch(url)
    .then(response => response.json())  
    .then(j => { //j={"(17,361,353,49)": [["endbar", [0, 0]], ["tine", [0, 1]], ["ine", [0, 2]], ["Lontadus", [0, 3]]], "(145,244,220,109)": [["image", [0, 0]], ["is", [0, 1]], ["sont", [0, 2]], ["Maim", [0, 3]], ["bat", [1, 0]], ["bat", [1, 1]], ["the", [1, 2]], ["maw", [1, 3]], ["thar", [2, 0]], ["gang", [2, 1]], ["to", [2, 2]], ["be", [2, 3]], ["third", [3, 0]]], "(140,90,229,149)": [["image", [0, 0]]], "(11,91,113,266)": [["sidebar", [0, 0]], ["rest", [1, 0]], ["calder", [2, 0]], ["peds", [3, 0]], ["beignet", [4, 0]], ["moment", [5, 0]]], "(9,4,362,72)": [["navbar", [0, 0]], ["Mome", [0, 1]], ["about", [0, 2]], ["logout", [0, 3]]]}
        
    console.log(j);
        // j={"slideshow"}
        this.setState({ data: j }, () => {
            this.colorpicker();
            document.getElementById("prev").style.background="#f2ec4c";
            this.call();
        });
        
        })
    

       
    }
    colorpicker()
    {
        var ci1=document.createElement("input")
        ci1.type="color";
        ci1.id="ci1"
        ci1.value="#f2ec4c"
        document.getElementById("download").appendChild(ci1);
        ci1.oninput=(e)=>
        {   
            document.getElementById("prev").style.background=e.target.value;
        }
        var ci2=document.createElement("input")
        ci2.type="color";
        ci2.value="#F2994A"
        ci2.id="ci2"
        document.getElementById("download").appendChild(ci2);
        ci2.oninput=(e)=>
        {   
            for (var i in this.state.side_state)
            {
                document.getElementById("pside"+i).style.background=e.target.value;
            }
            for (var i in this.state.nav_state)
            {
                document.getElementById("pnav"+i).style.background=e.target.value;
            }
            for (var i in this.state.end_state)
            {
                document.getElementById("pend"+i).style.background=e.target.value;
            }
            for (var i=0;i<Object.keys(this.state.data_ss).length;i++)
            {
                document.getElementById("slideshow_left"+i).style.background=e.target.value;
                document.getElementById("slideshow_right"+i).style.background=e.target.value;
            }

        }
        
    }
    ustate()
    {
        console.log(this.state);
    }
    placeDiv(x_pos, y_pos) {
        var d = document.getElementById('yourDivId');
        d.style.position = "absolute";
        d.style.left = x_pos+'px';
        d.style.top = y_pos+'px';
      }
    resiz(p)
    {
        var x = parseInt(p[0])
        x=((x)*(3000))/(1076)
        var w = parseInt(p[2])
        w=((w)*(3000))/(1076)
        var y = parseInt(p[1])
        y=((y)*(1400))/(759)
        var h = parseInt(p[3])
        h=((h)*(1400))/(759)
        if(x+w > window.innerWidth-(window.innerWidth*0.25))
            {
                console.log("errror")
                w=parseInt( window.innerWidth-(window.innerWidth*0.25)-50);
            }
        return [ Math.floor(x), Math.floor(y), Math.floor(w), Math.floor(h)]
    }
    final_resiz(p)
    {
        var x = parseInt(p[0])
        x=((x)*(3000))/(1076)
        var w = parseInt(p[2])
        w=((w)*(3000))/(1076)
        var y = parseInt(p[1])
        y=((y)*(1400))/(759)
        var h = parseInt(p[3])
        h=((h)*(1400))/(759)
        return [ Math.floor(x), Math.floor(y), Math.floor(w), Math.floor(h)]
    }
    call()
    {
        document.getElementById("filename").value="filename"
        console.log("call");
        var s_show=[];
        var d=this.state.data;
        for (var i in d)
         {  
            if(d[i][0][0]==="slideshow")
            {
                this.slideshow(d[i],i,this.state.no_of_slideshow+1)
                this.state.no_of_slideshow+=1;
            }
            else if(d[i][0][0]==="image")
                this.image(d[i],i);
            else if(d[i][0][0]==="sidebar")
                this.sidebar(d[i],i);
            else if(d[i][0][0]==="endbar")
                this.endbar(d[i],i);
            else if(d[i][0][0]==="navbar")
                this.navbar(d[i],i);
            else 
                this.text(d[i],i);          
        }
    }
    handleDownload = (url, filename) => {
        axios.get(url, {
          responseType: 'blob',
        })
        .then((res) => {
          fileDownload(res.data, filename)
        })
      }
    handleFileSelect(evt) {
        var files = evt.target.files;
        // var f = files[0];
        console.log(files)
        // var reader = new FileReader();
        // reader.onload = (function(theFile) {
        //     return function(e) {
        //         document.getElementById("eslide").val( e.target.result );
        //     };
        //   })(f);
        //   reader.readAsText(f);
      }

    slideshow(d,p,no)
    {
        
        var obj = document.createElement('div');
        obj.id = "slideshow"+Object.keys(this.state.data_ss).length.toString();
        this.state.data_ss["slideshow"+Object.keys(this.state.data_ss).length.toString()] = [] ;
        this.state.pres_ss["slideshow"+(Object.keys(this.state.data_ss).length-1).toString()] = 0 ;
        var pos= p.split("(")[1].split(")")[0].split(",");
        pos = this.resiz(pos);
        var pt = "position:absolute;top:"+pos[1]+"px;left:"+pos[0]+"px;width:"+pos[2]+"px;height:"+pos[3]+"px;border-style: groove;";
        obj.style.cssText = pt;

        var l_r =()=>
        {
        var p =document.getElementById("prev");
        var left = document.createElement("button")
        left.innerHTML="<"
        left.id = "slideshow_left"+(Object.keys(this.state.data_ss).length-1).toString()
        left.style.cssText="position:absolute;width:20px;height:"+pos[3]+"px;left:0px;border-radius: 8px;background-color: #F2994A;";
        left.onclick = ()=>{
                if(this.state.pres_ss[obj.id] > 1)
                {
                    this.state.pres_ss[obj.id] =this.state.pres_ss[obj.id]-1;
                    if(document.getElementById("s_img"+obj.id.toString()))
                        document.getElementById("s_img"+obj.id.toString()).remove();
                    var s_img=document.createElement("img");
                    s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:20px;"
                    s_img.id="s_img"+obj.id.toString();
                    s_img.src=this.state.data_ss[obj.id][this.state.pres_ss[obj.id]-1];
                    obj.appendChild(s_img);
                }
                else
                {
                    this.state.pres_ss[obj.id] =this.state.data_ss[obj.id].length;
                    if(document.getElementById("s_img"+obj.id.toString()))
                        document.getElementById("s_img"+obj.id.toString()).remove();

                    var s_img=document.createElement("img");
                    s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:20px;"
                    s_img.id="s_img"+obj.id.toString();
                    s_img.src=this.state.data_ss[obj.id][this.state.pres_ss[obj.id]-1];
                    obj.appendChild(s_img);
                }
                console.log(this.state.pres_ss[obj.id])
        
        };
        
        obj.appendChild(left);
        
        var right = document.createElement("button")
        right.innerHTML=">";
        right.id = "slideshow_right"+(Object.keys(this.state.data_ss).length-1).toString();
        right.style.cssText="position:absolute;width:20px;height:"+pos[3]+"px;right:0px;border-radius: 8px;background-color: #F2994A;";
        right.onclick = ()=>
        {
            if(this.state.pres_ss[obj.id] < this.state.data_ss[obj.id].length)
            {
                this.state.pres_ss[obj.id] =this.state.pres_ss[obj.id]+1;
                if(document.getElementById("s_img"+obj.id.toString()))
                        document.getElementById("s_img"+obj.id.toString()).remove();

                var s_img=document.createElement("img");
                s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:20px;"
                s_img.id="s_img"+obj.id.toString();
                s_img.src=this.state.data_ss[obj.id][this.state.pres_ss[obj.id]-1];
                obj.appendChild(s_img);
                
            }
            else
            {
                this.state.pres_ss[obj.id] =1;
                if(document.getElementById("s_img"+obj.id.toString()))
                        document.getElementById("s_img"+obj.id.toString()).remove();

                var s_img=document.createElement("img");
                s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:20px;"
                s_img.id="s_img"+obj.id.toString();
                s_img.src=this.state.data_ss[obj.id][this.state.pres_ss[obj.id]-1];
                obj.appendChild(s_img)
            }
            console.log(this.state.pres_ss[obj.id])
    
        };
        obj.appendChild(right);
    
        }
        
        var s_img=document.createElement("img");
        s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:20px;"
        s_img.id="s_img"+obj.id.toString();
        document.getElementById("prev").appendChild(obj);
        l_r();
        

        obj=document.getElementById("slideshow"+(Object.keys(this.state.data_ss).length-1).toString())
        console.log(this.state,"slideshow"+(Object.keys(this.state.data_ss).length-1).toString())
        
        obj.onclick =() =>{
            
            const myNode = document.getElementById("edit");
            while (myNode.firstChild) {
                myNode.removeChild(myNode.lastChild);
            }
            var h = document.createElement('h1');
            h.innerText="click on plus to add images to your sildeshow"
            document.getElementById("edit").appendChild(h);
            
        var add_b = document.createElement('button');
        add_b.id = "e_b";
        add_b.innerHTML="+"
        // var no_of_input=0;
        add_b.onclick  =() =>{
            var new_input=document.createElement("input");
            new_input.id="eb"+this.state.data_ss[obj.id].length.toString()
            document.getElementById("edit").appendChild(new_input);
            this.state.data_ss[obj.id].push("blank")
            // no_of_input=no_of_input+1
        }
        add_b.style.cssText="position:absolute;left:0px;bottom:40%"
        document.getElementById("edit").appendChild(add_b);

        var del_b = document.createElement('button');
        del_b.id = "e_b";
        del_b.innerHTML="-"
        del_b.onclick  =() =>
        {
        
            if(this.state.data_ss[obj.id].length != 0)
            {
            this.state.data_ss[obj.id].pop()
            document.getElementById("eb"+this.state.data_ss[obj.id].length.toString()).remove();
            }
            
        }
        del_b.style.cssText="position:absolute;left:20px;bottom:40%"
        document.getElementById("edit").appendChild(del_b);

        //saved display
        for (var i in this.state.data_ss[obj.id])
        {
            var new_input=document.createElement("input");
            new_input.id="eb"+i.toString()
            new_input.value=this.state.data_ss[obj.id][i]
            // no_of_input=this.state.s_show.length
            document.getElementById("edit").appendChild(new_input);
        }
        

        var ref_b = document.createElement('button');
        ref_b.id = "e_b";
        ref_b.innerHTML="SAVE changes"
        
        ref_b.onclick =  (e) =>
        {
            // while (obj.firstChild) 
            // {
            //     // console.log(obj.lastChild.id,"slideshow_right"+(Object.keys(this.state.data_ss).length-1).toString());
            //     // if(obj.lastChild.id!=="slideshow_right"+(Object.keys(this.state.data_ss).length-1).toString() && obj.lastChild.id!=="slideshow_left"+(Object.keys(this.state.data_ss).length-1).toString())
            //     obj.removeChild(obj.lastChild);
            // }
            // l_r();
            // var left = document.getElementById("slideshow_left"+(Object.keys(this.state.data_ss).length-1).toString());
            // console.log("s",left);
            // function bubble() {
            //     console.log('bubble: ' + this.firstChild.nodeValue.trim());
            // }
            // left.addEventListener('click', bubble, false);
            // left.onclick = function(){console.log("s");};
            // left.onClick = ()=>
            // {
                
            //     console.log("s");
            //     // if(this.state.pres_ss["slideshow"+Object.keys(this.state.data_ss).length.toString()] > 1)
            //     //     this.state.pres_ss["slideshow"+Object.keys(this.state.data_ss).length.toString()] =this.state.pres_ss["slideshow"+Object.keys(this.state.data_ss).length.toString()]-1;
            //     // else
            //     //     this.state.pres_ss["slideshow"+Object.keys(this.state.data_ss).length.toString()] =Object.keys(this.state.data_ss).length;
            //     // console.log(this.state.pres_ss["slideshow"+Object.keys(this.state.data_ss).length.toString()])
            // }
            console.log(this.state.data_ss[obj.id]);
            if(this.state.data_ss[obj.id].length !== 0)
            {
                // this.getState((curState) => {console.log(curState)});
                var b =this.state.data_ss[obj.id];
                this.state.data_ss[obj.id]=[];
                this.state.pres_ss[obj.id]=0;
                
                for(var i=0;i<b.length;i++)
                {
                    // var o =window.location.origin.toString() + document.getElementById("eb"+i.toString()).value;
                    // var s_img_link=document.getElementById("eb"+i.toString()).value;
                    this.state.data_ss[obj.id].push(document.getElementById("eb"+i.toString()).value);
                    this.state.pres_ss[obj.id]=this.state.pres_ss[obj.id]+1;
                    // var s_img=document.createElement("img");
                    // // import im1 from s_img_link;
                    // s_img.src=s_img_link;
                    // s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:21px;"
                    // obj.appendChild(s_img);
                    

                }
                if(document.getElementById("s_img"+obj.id.toString()))
                    document.getElementById("s_img"+obj.id.toString()).remove();

                var s_img=document.createElement("img");
                s_img.style.cssText="position:absolute;width:"+(pos[2]-40).toString()+"px;height:"+(pos[3]).toString()+"px;left:20px;"
                s_img.id="s_img"+obj.id.toString();
                s_img.src=this.state.data_ss[obj.id][this.state.pres_ss[obj.id]-1];
                obj.appendChild(s_img);
                console.log(this.state.pres_ss[obj.id]-1);
            }          
        }
        ref_b.style.cssText="position:absolute;left:50px;bottom:40%"
        document.getElementById("edit").appendChild(ref_b);
        
        
    };
        

    }
    image(d,p)
    {
        
        var obj = document.createElement('img');
        var pos= p.split("(")[1].split(")")[0].split(",");
        pos = this.resiz(pos);
        console.log(pos)
        var pt = "position:absolute;top:"+pos[1]+"px;left:"+pos[0]+"px;width:"+pos[2]+"px;height:"+pos[3]+"px;border-style: groove;";
        obj.style.cssText = pt;
        // Object.keys(this.state.data_img).length.toString()
        obj.id="img"+Object.keys(this.state.data_img).length.toString()
        this.state.data_img["img"+Object.keys(this.state.data_img).length.toString()]=""
        document.getElementById("prev").appendChild(obj);
        obj.onclick = ()=>
        {
            const myNode = document.getElementById("edit");
            while (myNode.firstChild) {
                myNode.removeChild(myNode.lastChild);
            }
            var x=document.createElement("input");
            x.value=this.state.data_img[obj.id];
            x.onchange = ()=>{this.state.data_img[obj.id]=x.value}
            document.getElementById("edit").appendChild(x);
            var ref_b = document.createElement('button');
        ref_b.id = "e_b";
        ref_b.innerHTML="SAVE changes"
        
        ref_b.onclick =  () =>
        {
            obj.src=this.state.data_img[obj.id];
        }
        document.getElementById("edit").appendChild(ref_b);

        }
        

        
    }
    sidebar(d,p)
    {   console.log(p,"pppp")
        for (var i in d)
        {
            if(i!=0){
                this.state.side_state.push(d[i][0]);
                this.state.lside_state.push("blank");
            }
        }
        
        var obj = document.createElement('div');
        var pos= p.split("(")[1].split(")")[0].split(",");
        pos = this.resiz(pos);
        var side_inner_box_width= parseInt( window.innerWidth*0.15);
        // pos[3]=parseInt( window.innerHeight-(2*(window.innerHeight*(0.1))));
        var pt = "position:absolute;top:"+parseInt(window.innerHeight*0.1)+"px;left:"+0+"px;width:"+side_inner_box_width+"px;height:"+pos[3]+"px;border-style: groove;overflow:hidden";
        obj.style.cssText = pt;
        obj.id="side"
        document.getElementById("prev").appendChild(obj);
        
        // 3
        var disp = ()=>
        {
            const myNode = document.getElementById("side");
        while (myNode.firstChild) {
            myNode.removeChild(myNode.lastChild);
        }
        var t = Math.floor(parseInt(pos[3])/(this.state.side_state.length))
        for (var i in this.state.side_state)
        {
            var x = document.createElement("div")
            x.id="pside"+i.toString()
            obj.appendChild(x)
            var tt=Math.floor(parseInt(t)/2)-10;
            x.style.cssText="position:absolute;width:"+side_inner_box_width+"px;top:"+((i)*t)+"px;height:"+t+"px;background-color:"+document.getElementById("ci2").value+";text-align: center;padding-top:"+tt+"px;border-style: groove;"
            x.innerHTML=this.state.side_state[i]
         }
        }
        var save_ = ()=>
        {
            if(this.state.side_state.length !== 0)
                {
                    console.log("curState");
                    var l=this.state.side_state.length;
                    this.setState({ side_state: [] });
                    this.setState({ lside_state: [] });
                    for(var i=0;i<l;i++)
                    {
                        this.state.side_state.push(document.getElementById("side"+i.toString()).value)
                        this.state.lside_state.push(document.getElementById("lside"+i.toString()).value);
                       
                        if(this.state.side_state[i]!=="")
                        document.getElementById("lside"+i.toString()).placeholder="LINK for "+this.state.side_state[i];
                        else
                        document.getElementById("lside"+i.toString()).placeholder="<-Tip: Enter name first";
                        if(this.state.lside_state[i]!=="blank")
                                document.getElementById("lside"+i.toString()).value=this.state.lside_state[i];                   
                    }
                    console.log(this.state.side_state);
                } 
                disp();
        }  
        disp()
        // 4
        obj.onclick = ()=>
        {

            const myNode = document.getElementById("edit");
            while (myNode.firstChild) {
                myNode.removeChild(myNode.lastChild);
            }
            for (var i in this.state.side_state)
            {

                    var x = document.createElement("input")
                    document.getElementById("edit").appendChild(x)
                    x.value=this.state.side_state[i]
                    x.id="side"+(i).toString();

                    var x1 = document.createElement("input")
                    if(this.state.side_state[i]!=="")
                            x1.placeholder="LINK for "+this.state.side_state[i];
                    else
                            x1.placeholder="<-Tip: Enter name first";
                    if(this.state.lside_state[i]!=="blank")
                            x1.value=this.state.lside_state[i];
                    document.getElementById("edit").appendChild(x1)
                    x1.id="lside"+(i).toString();
                    x.onkeyup=(ev)=>{
                        save_();
                        disp();
                    }
                    x1.onkeyup=(ev)=>{
                        save_();
                    }
             }
            // 5
            var add_b = document.createElement('button');
            add_b.id = "e_b";
            add_b.innerHTML="+"
            add_b.style.cssText="position:absolute;left:0px;bottom:40%"

            add_b.onclick =  (e) =>
            {
                var new_input=document.createElement("input");
                new_input.id="side"+this.state.side_state.length.toString()
                this.state.side_state.push("side"+this.state.side_state.length)
                document.getElementById("edit").appendChild(new_input);

                var new_input1=document.createElement("input");
                new_input1.id="lside"+this.state.lside_state.length.toString();
                this.state.lside_state.push("");
                new_input1.placeholder="<-Tip: Enter name first";
                document.getElementById("edit").appendChild(new_input1);

                save_();
                disp();

                new_input.onkeyup=(ev)=>{
                    save_();
                    disp();
                }
                new_input1.onkeyup=(ev)=>{
                    save_();
                }

            }
            document.getElementById("edit").appendChild(add_b);

            var del_b = document.createElement('button');
            del_b.id = "e_b";
            del_b.innerHTML="-"
            del_b.style.cssText="position:absolute;left:20px;bottom:40%"
            del_b.onclick =  (e) =>
            {
                if(this.state.side_state.length != 0)
                {
                    this.state.side_state.pop();
                    document.getElementById("side"+(this.state.side_state.length).toString()).remove();
                    this.state.lside_state.pop();
                    document.getElementById("lside"+(this.state.lside_state.length).toString()).remove();
                } 
                save_();
                disp();
            }
            document.getElementById("edit").appendChild(del_b);


        //     var ref_b = document.createElement('button');
        //     ref_b.id = "e_b";
        //     ref_b.innerHTML="SAVE changes"
        //     ref_b.style.cssText="position:absolute;left:50px;bottom:40%"
        //     ref_b.onclick =  (e) =>
        //     {
        //         if(this.state.side_state.length !== 0)
        //         {
        //             console.log("curState");
        //             var l=this.state.side_state.length;
        //             this.setState({ side_state: [] });
        //             this.setState({ lside_state: [] });
        //             for(var i=0;i<l;i++)
        //             {
        //                 this.state.side_state.push(document.getElementById("side"+i.toString()).value)
        //                 this.state.lside_state.push(document.getElementById("lside"+i.toString()).value);
                       
        //                 if(this.state.side_state[i]!=="")
        //                 document.getElementById("lside"+i.toString()).placeholder="LINK for "+this.state.side_state[i];
        //                 else
        //                 document.getElementById("lside"+i.toString()).placeholder="<-Tip: Enter name first";
        //                 if(this.state.lside_state[i]!=="blank")
        //                         document.getElementById("lside"+i.toString()).value=this.state.lside_state[i];                   
        //             }
        //             console.log(this.state.side_state);
        //         } 
        //         disp();         
        //     }
        
        // document.getElementById("edit").appendChild(ref_b);
        }


        
        
    }
    endbar(d,p)
    {
        // 1
        for (var i in d)
        {
            if(i!=0){
            this.state.end_state.push(d[i][0]);
            this.state.lend_state.push("blank");
            }
        }
        // 2
        var obj = document.createElement('div');
        var pos= p.split("(")[1].split(")")[0].split(",");
        pos = this.resiz(pos);
        pos[2]=window.innerWidth-(window.innerWidth*0.25);
        var h=parseInt(window.innerHeight*0.1)
        // var pt = "position:absolute;top:"+0+"px;left:"+0+"px;width:"+(window.innerWidth-(window.innerWidth*0.25)).toString()+"px;height:"+pos[3]+"px;border-style: solid;overflow:hidden";
        var pt = "position:absolute;px;left:"+0+"px;width:"+(window.innerWidth-(window.innerWidth*0.25)).toString()+"px;height:"+h+"px;border-style: groove;overflow:hidden;bottom:0px";
        obj.style.cssText = pt;
        obj.id="end"
        document.getElementById("prev").appendChild(obj);
        // 3
        var disp = ()=>
        {
            const myNode = document.getElementById("end");
        while (myNode.firstChild) {
            myNode.removeChild(myNode.lastChild);
        }
        var t = Math.floor(parseInt(pos[2])/(this.state.end_state.length))
        console.log("t",t,this.state.end_state.length,pos[2])
        for (var i in this.state.end_state)
         {
            var x = document.createElement("div")
            x.id="pend"+i.toString()
            obj.appendChild(x)
            var tt=Math.floor(h/2)-10
            x.style.cssText="position:absolute;width:"+t+"px;left:"+((i)*t)+"px;height:"+h+"px;background-color:"+document.getElementById("ci2").value+";text-align: center;padding-top:"+tt+"px;border-style: groove;"
            x.innerHTML=this.state.end_state[i]
            
         }
        }
        var save_= ()=>
        {
            if(this.state.end_state.length !== 0)
                {
                    console.log("curState");
                    var l=this.state.end_state.length;
                    this.setState({ end_state: [] });
                    this.setState({ lend_state: [] });
                    for(var i=0;i<l;i++)
                    {
                        console.log("end"+i.toString());
                        this.state.end_state.push(document.getElementById("end"+i.toString()).value);   
                        this.state.lend_state.push(document.getElementById("lend"+i.toString()).value);
                       
                        if(this.state.end_state[i]!=="")
                        document.getElementById("lend"+i.toString()).placeholder="LINK for "+this.state.end_state[i];
                        else
                        document.getElementById("lend"+i.toString()).placeholder="<-Tip: Enter name first";
                       if(this.state.lend_state[i]!=="blank")
                            document.getElementById("lend"+i.toString()).value=this.state.lend_state[i];                
                    }
                    console.log(this.state.end_state,l);
                }
        }
        disp();
        // 4
        obj.onclick = ()=>
        {

            const myNode = document.getElementById("edit");
            while (myNode.firstChild) {
                myNode.removeChild(myNode.lastChild);
            }
            for (var i in this.state.end_state)
            {
                    var x = document.createElement("input")
                    document.getElementById("edit").appendChild(x)
                    x.value=this.state.end_state[i]
                    x.id="end"+(i).toString();

                    var x1 = document.createElement("input")
                    if(this.state.end_state[i]!=="")
                            x1.placeholder="LINK for "+this.state.end_state[i];
                    else
                            x1.placeholder="<-Tip: Enter name first";
                    if(this.state.lend_state[i]!=="blank")
                            x1.value=this.state.lend_state[i];
                    document.getElementById("edit").appendChild(x1)
                    x1.id="lend"+(i).toString();
                    x.onkeyup=(ev)=>{
                        save_();
                        disp();
                    }
                    x1.onkeyup=(ev)=>{
                        save_();
                    }
             }
            // 5
            var add_b = document.createElement('button');
            add_b.id = "e_b";
            add_b.innerHTML="+"
            add_b.style.cssText="position:absolute;left:0px;bottom:40%"
            
            add_b.onclick =  (e) =>
            { 
                var new_input=document.createElement("input");
                new_input.id="end"+this.state.end_state.length.toString()
                this.state.end_state.push("end"+this.state.end_state.length)
                document.getElementById("edit").appendChild(new_input);

                var new_input1=document.createElement("input");
                new_input1.id="lend"+this.state.lend_state.length.toString();
                this.state.lend_state.push("");
                new_input1.placeholder="<-Tip: Enter name first";
                document.getElementById("edit").appendChild(new_input1);

                save_();
                disp();

                new_input.onkeyup=(ev)=>{
                    save_();
                    disp();
                }
                new_input1.onkeyup=(ev)=>{
                    save_();
                }
                
            }
            document.getElementById("edit").appendChild(add_b);

            var del_b = document.createElement('button');
            del_b.id = "e_b";
            del_b.innerHTML="-"
            del_b.style.cssText="position:absolute;left:20px;bottom:40%"
            del_b.onclick =  (e) =>
            {
                console.log(this.state.end_state);
                if(this.state.end_state.length != 0)
                {
                    this.state.end_state.pop();
                    document.getElementById("end"+(this.state.end_state.length).toString()).remove();
                    this.state.lend_state.pop();
                    document.getElementById("lend"+(this.state.lend_state.length).toString()).remove();
                }
                save_();
                disp(); 
            }
            document.getElementById("edit").appendChild(del_b);


        //     var ref_b = document.createElement('button');
        //     ref_b.id = "e_b";
        //     ref_b.innerHTML="SAVE changes"
        //     ref_b.style.cssText="position:absolute;left:50px;bottom:40%"
            
        //     ref_b.onclick =  (e) =>
        //     {
        //         if(this.state.end_state.length !== 0)
        //         {
        //             console.log("curState");
        //             var l=this.state.end_state.length;
        //             this.setState({ end_state: [] });
        //             this.setState({ lend_state: [] });
        //             for(var i=0;i<l;i++)
        //             {
        //                 console.log("end"+i.toString());
        //                 this.state.end_state.push(document.getElementById("end"+i.toString()).value);   
        //                 this.state.lend_state.push(document.getElementById("lend"+i.toString()).value);
                       
        //                 if(this.state.end_state[i]!=="")
        //                 document.getElementById("lend"+i.toString()).placeholder="LINK for "+this.state.end_state[i];
        //                 else
        //                 document.getElementById("lend"+i.toString()).placeholder="<-Tip: Enter name first";
        //                if(this.state.lend_state[i]!=="blank")
        //                     document.getElementById("lend"+i.toString()).value=this.state.lend_state[i];                
        //             }
        //             console.log(this.state.end_state,l);
        //         }   
        //     disp();         
        // }
        
        // document.getElementById("edit").appendChild(ref_b);
        }

        


    }
    navbar(d,p)
    {
       // 1
       for (var i in d)
       {
           if(i!=0)
           {
           this.state.nav_state.push(d[i][0]);
           this.state.lnav_state.push("blank");
           }
       }
       // 2

       var obj = document.createElement('div');
       var pos= p.split("(")[1].split(")")[0].split(",");
       pos = this.resiz(pos);
       pos[2]=window.innerWidth-(window.innerWidth*0.25);
       var h=parseInt(window.innerHeight*0.1)
       var pt = "position:fixed;top:"+0+"px;left:"+0+"px;width:"+(window.innerWidth-(window.innerWidth*0.25)).toString()+"px;height:"+h+"px;border-style: groove;overflow:hidden";
       obj.style.cssText = pt;
       obj.id="nav"
       document.getElementById("prev").appendChild(obj);
       // 3
       var disp = ()=>
       {
        const myNode = document.getElementById("nav");
        while (myNode.firstChild) {
            myNode.removeChild(myNode.lastChild);
        }
       var t = Math.floor(parseInt(pos[2])/(this.state.nav_state.length))
       console.log("t",t,this.state.nav_state.length,pos[2])
       for (var i in this.state.nav_state)
        {
        
           var x = document.createElement("div")
           x.id="pnav"+i.toString()
           obj.appendChild(x)
           var tt=Math.floor(h/2)-10
           x.style.cssText="position:absolute;width:"+t+"px;left:"+((i)*t)+"px;height:"+h+"px;background-color:"+document.getElementById("ci2").value+";text-align: center;padding-top:"+tt+"px;border-style: groove;"
           x.innerHTML=this.state.nav_state[i]
           console.log("x"+x.toString());
        }
       }
       var save_ = ()=>
       {
        if(this.state.nav_state.length !== 0)
        {
            console.log("curState");
            var l=this.state.nav_state.length;
            this.setState({ nav_state: [] });
            this.setState({ lnav_state: [] });
            for(var i=0;i<l;i++)
            {
                console.log("nav"+i.toString());
                this.state.nav_state.push(document.getElementById("nav"+i.toString()).value);
                this.state.lnav_state.push(document.getElementById("lnav"+i.toString()).value);
                
                 if(this.state.nav_state[i]!=="")
                 document.getElementById("lnav"+i.toString()).placeholder="LINK for "+this.state.nav_state[i];
                 else
                 document.getElementById("lnav"+i.toString()).placeholder="<-Tip: Enter name first";
                if(this.state.lnav_state[i]!=="blank")
                     document.getElementById("lnav"+i.toString()).value=this.state.lnav_state[i];
                                 
            }
            console.log(this.state.nav_state,l);
         } 
       }
       disp();
       // 4
       obj.onclick = ()=>
       {

           const myNode = document.getElementById("edit");
           while (myNode.firstChild) {
               myNode.removeChild(myNode.lastChild);
           }
           for (var i in this.state.nav_state)
           {
                   var x = document.createElement("input")
                   document.getElementById("edit").appendChild(x)
                   x.value=this.state.nav_state[i];
                   x.id="nav"+(i).toString();

                   var x1 = document.createElement("input")
                   if(this.state.nav_state[i]!=="")
                        x1.placeholder="LINK for "+this.state.nav_state[i];
                   else
                        x1.placeholder="<-Tip: Enter name first";
                   if(this.state.lnav_state[i]!=="blank")
                        x1.value=this.state.lnav_state[i];
                   document.getElementById("edit").appendChild(x1)
                   x1.id="lnav"+(i).toString();

                   x.onkeyup=(ev)=>{
                    save_();
                    disp();
                    }
                    x1.onkeyup=(ev)=>{
                        save_();
                    }
            }
           // 5
           var add_b = document.createElement('button');
           add_b.id = "e_b";
           add_b.innerHTML="+"
           add_b.style.cssText="position:absolute;left:0px;bottom:40%"
           add_b.onclick =  (e) =>
           { 
               var new_input=document.createElement("input");
               new_input.id="nav"+this.state.nav_state.length.toString()
               this.state.nav_state.push("nav"+this.state.nav_state.length)
               document.getElementById("edit").appendChild(new_input);

               var new_input1=document.createElement("input");
               new_input1.id="lnav"+this.state.lnav_state.length.toString();
               this.state.lnav_state.push("");
               new_input1.placeholder="<-Tip: Enter name first";
               document.getElementById("edit").appendChild(new_input1);
               save_();
                disp();

               new_input.onkeyup=(ev)=>{
                save_();
                disp();
                }
                new_input1.onkeyup=(ev)=>{
                    save_();
                }

           }
           document.getElementById("edit").appendChild(add_b);

           var del_b = document.createElement('button');
           del_b.id = "e_b";
           del_b.innerHTML="-"
           del_b.style.cssText="position:absolute;left:20px;bottom:40%"
           del_b.onclick =  (e) =>
           {
               console.log(this.state.nav_state);
               if(this.state.nav_state.length != 0)
               {
                   this.state.nav_state.pop();
                   document.getElementById("nav"+(this.state.nav_state.length).toString()).remove();
                   this.state.lnav_state.pop();
                   document.getElementById("lnav"+(this.state.lnav_state.length).toString()).remove();
               }
               save_();
                disp();
           }
           document.getElementById("edit").appendChild(del_b);


       }
    }
    text(d,p)
    {
        console.log("text",p)
        var obj = document.createElement('div');
        obj.id="text"+this.state.data_text.toString();
        this.state.data_text=this.state.data_text+1;
        var pos= p.split("(")[1].split(")")[0].split(",");
        pos = this.resiz(pos);
        console.log(pos[2])
        if((parseInt(pos[2])+parseInt(pos[0]))>(0.75)*(window.innerWidth))
        {    
            console.log("sss",0.9*(window.innerWidth-parseInt(pos[0])))
            var pos_2=0.9*((0.75*window.innerWidth)-parseInt(pos[0])).toString();
            var pt = "position:absolute;top:"+pos[1]+"px;left:"+pos[0]+"px;width:"+pos_2+"px;height:"+pos[3]+"px;display: inline-block;";
        }
        else
        {
            var pt = "position:absolute;top:"+pos[1]+"px;left:"+pos[0]+"px;width:"+pos[2]+"px;height:"+pos[3]+"px;display: inline-block;";
        }
        console.log(pos[2])
        obj.style.cssText = pt;
        var t ="";
        document.getElementById("prev").appendChild(obj);
        for (var i in d)
        { 
            if(i!=0)
            {
            console.log(i,"texttt")
            t = t+d[i][0]+" ";
            }
        }
         var x = document.createTextNode(t);
         obj.appendChild(x)
         obj.onclick = function(ele) {
            this.state = { first: 0 };
            const myNode = document.getElementById("edit");
            while (myNode.firstChild) {
                myNode.removeChild(myNode.lastChild);
            }
        var e_obj = document.createElement('input');
        e_obj.type = "text";
        e_obj.id = "etext";
        if(this.state.first)
            e_obj.value =t;
        else
            e_obj.value =obj.innerHTML;
        e_obj.onkeyup = function (e)
        {
            obj.innerHTML=e.target.value
        }
        var font = ["Times New Roman, Times, serif;" , "Arial, Helvetica, sans-serif;","Lucida Console, Courier New, monospace;"]
        var selectElement = document.createElement ("select");
        for (var i=0;i < font.length;i++)
            {
                var option = new Option (font[i], "Value" + font[i]);
                selectElement.options[selectElement.options.length] = option;
            }
            document.getElementById("edit").appendChild(selectElement);
        selectElement.onchange = function(e)
        {
            
            obj.style.cssText=pt+"font-family:"+e.target.value;
        }
        
        
        
        document.getElementById("edit").appendChild(e_obj);
        
    };
    
   }
   showAlert=() => {
       console.log(JSON.stringify(this.state.data_ss));
       console.log(JSON.stringify(this.state.pres_ss));
        // console.log(this.state)
        var final = "";
         var all = document.getElementById("prev").getElementsByTagName("*");
         
        for (var i=0; i < all.length; i++) 
        {
            if (all[i].id ==="nav")
            {   
        
                var t = parseInt(window.innerWidth/(this.state.nav_state.length));
                var final_nav=document.createElement("div");
                final_nav.id="final_nav";
                final_nav.style.cssText=all[i].style.cssText+"width:100%";
                console.log(this.state.lnav_state)
                var c=this.state.lnav_state.length;
                for(var child=0; child<c; child++) 
                {
                    var hd1=document.getElementById("pnav"+child.toString());
                    var hd = hd1.cloneNode(true)
                    var l=document.createElement("a");
                    l.href=this.state.lnav_state[child];
                    hd.innerHTML=this.state.nav_state[child];
                    l.style.cssText="color: inherit;text-decoration: none !important"
                    hd.style.cssText=hd.style.cssText+"width:"+t+"px;left:"+((child)*t)+"px;vertical-align: middle;";
                    l.appendChild(hd);
                    final_nav.appendChild(l)
                }
                console.log("c",final_nav,t);
                final+= final_nav.outerHTML;
            }
            
            if(all[i].id==="side")
            {
                var final_slide=document.createElement("div");
                final_slide.id="final_side";
                final_slide.style.cssText=all[i].style.cssText;
                console.log(this.state.lside_state)
                var c=this.state.lside_state.length;
                var final_inner_width= parseInt((all[i].getBoundingClientRect().width-5));
                var final_inner_height= parseInt((all[i].getBoundingClientRect().height-5)/c);
                for(var child=0; child<c; child++) 
                {
                    var hd1=document.getElementById("pside"+child.toString());
                    var hd = hd1.cloneNode(true)
                    var l=document.createElement("a");
                    l.href=this.state.lside_state[child];
                    l.style.cssText="color: inherit;text-decoration: none !important;height:"+final_inner_height.toString()+"px;width:"+final_inner_width.toString()+"px;top:"+(child*final_inner_height);
                    hd.style.cssText+=";height:"+final_inner_height.toString()+"px;width:"+final_inner_width.toString()+"px;top:"+(child*final_inner_height);
                    hd.innerHTML=this.state.side_state[child];
                    // var final_side_new=document.createElement("div");
                    // final_side_new.style.cssText=hd.style
                    console.log(hd.getBoundingClientRect().width,"l")
                    hd.style.cssText=hd.style.cssText+"vertical-align: middle;";
                    l.appendChild(hd);
                    final_slide.appendChild(l)
                    

                }
                // console.log("c",final_slide);
                final+= final_slide.outerHTML;
            }
            if(all[i].id==="end")
            {
                var t = parseInt(window.innerWidth/(this.state.end_state.length));
                var final_end=document.createElement("div");
                final_end.id="final_end";
                final_end.style.cssText=all[i].style.cssText+"width:100%;top:990px";
                console.log(this.state.lend_state)
                var c=this.state.lend_state.length;
                for(var child=0; child<c; child++) 
                {
                    var hd1=document.getElementById("pend"+child.toString());
                    var hd = hd1.cloneNode(true)
                    var l=document.createElement("a");
                    l.href=this.state.lend_state[child];
                    hd.innerHTML=this.state.end_state[child];
                    l.style.cssText="color: inherit;text-decoration: none !important"
                    hd.style.cssText=hd.style.cssText+"width:"+t+"px;left:"+((child)*t)+"px;padding-top:0px;vertical-align: middle;line-height: 90px";
                    l.appendChild(hd);
                    final_end.appendChild(l)
                }
                // console.log("c",final_end);
                final+= final_end.outerHTML;
            }
            if(all[i].id.indexOf("text")!== -1)
            {
                all[i].style.cssText+=";width:"+(all[i].getBoundingClientRect().width+parseInt((all[i].getBoundingClientRect().width)*0.25)).toString()+"px;"
                all[i].style.cssText+=";left:"+(all[i].getBoundingClientRect().left+parseInt((all[i].getBoundingClientRect().left)*0.25)).toString()+"px;"
                // console.log("text",all[i])
                final+= all[i].outerHTML;
                
            }
            if(all[i].id.indexOf("slideshow")!== -1 && all[i].id.indexOf("left")=== -1 && all[i].id.indexOf("right")=== -1 && all[i].id.indexOf("s_img")=== -1)
            {
                var new_ss= document.createElement("div");
                // all[i].style.cssText+=";width:"+(all[i].getBoundingClientRect().width+parseInt((all[i].getBoundingClientRect().width)*0.25)).toString()+"px;"
                console.log("ss",all[i]);
                // document.getElementById("s_img"+all[i].id).style.cssText+=";width:"+(document.getElementById("s_img"+all[i].id).getBoundingClientRect().width+parseInt((document.getElementById("s_img"+all[i].id).getBoundingClientRect().width)*0.25)).toString()+"px;";
                new_ss.style.cssText=all[i].style.cssText+";width:"+(all[i].getBoundingClientRect().width+parseInt((all[i].getBoundingClientRect().width)*0.25)).toString()+"px;";
                new_ss.id=all[i].id;
                final+= new_ss.outerHTML;
                
            }

            if(all[i].id.indexOf("img")!== -1 && all[i].id.indexOf("slideshow")=== -1)
            {
                all[i].style.cssText+=";width:"+(all[i].getBoundingClientRect().width+parseInt((all[i].getBoundingClientRect().width)*0.25)).toString()+"px;"
                // console.log("text",all[i])
                final+= all[i].outerHTML;
            }

        }
        final=final+"</body></html>";//+"!@#$%^&*()var x="+JSON.stringify(this.state.data_ss)+";var y= "+JSON.stringify(this.state.pres_ss)
        console.log(final);


        var part1 ='<!DOCTYPE html><html lang="en"><head><script>\nvar x='+JSON.stringify(this.state.data_ss)+";var y= "+JSON.stringify(this.state.pres_ss)+";var clr='"+document.getElementById("ci2").value+"';\n";
        var part2= 'var h_w ={};</script></head><body style="height: 1000px;background: '+document.getElementById("prev").style.background+'; display:inline-block;"><script>window.onload =()=>{var c=0;for (const [key, value] of Object.entries(x)) {var h=document.getElementById(key).getBoundingClientRect().height-5;var w=document.getElementById(key).getBoundingClientRect().width-5;h_w[key]=[h,w];}function foo (i,key){document.getElementById(i).onclick = ()=>{if(y[key] > 1){w=document.getElementById(i).parentNode.getBoundingClientRect().width-5;h=document.getElementById(i).parentNode.getBoundingClientRect().height-5;y[key] =y[key]-1;if(document.getElementById("s_img"+key.toString())){document.getElementById("s_img"+key.toString()).remove();}var s_img=document.createElement("img");s_img.style.cssText="position:absolute;width:"+(w-40).toString()+"px;height:"+(h).toString()+"px;left:20px;";s_img.id="s_img"+key.toString();s_img.src=x[key][y[key]-1];document.getElementById(key).appendChild(s_img);}else{w=document.getElementById(i).parentNode.getBoundingClientRect().width-5;h=document.getElementById(i).parentNode.getBoundingClientRect().height-5;y[key] =x[key].length;if(document.getElementById("s_img"+key.toString())){document.getElementById("s_img"+key.toString()).remove();}var s_img=document.createElement("img");s_img.style.cssText="position:absolute;width:"+(w-40).toString()+"px;height:"+(h).toString()+"px;left:20px;";s_img.id="s_img"+key.toString();s_img.src=x[key][y[key]-1];document.getElementById(key).appendChild(s_img);}};}for (const [key, value] of Object.entries(x)) {h=h_w[key][0];w=h_w[key][1];var s_img=document.createElement("img");s_img.style.cssText="position:absolute;width:"+(w-40).toString()+"px;height:"+(h).toString()+"px;left:20px;";s_img.id="s_img"+key;s_img.src=x[key][y[key]-1];document.getElementById(key).append(s_img);var left = document.createElement("button");left.innerHTML="<";left.id = "slideshow_left"+(c).toString();left.style.cssText="position:absolute;width:20px;height:"+h+"px;left:0px;border-radius: 8px;background-color: "+clr+";";document.getElementById(key).appendChild(left);foo("slideshow_left"+(c).toString(),key);var right = document.createElement("button");right.innerHTML=">";right.id = "slideshow_right"+(c).toString();right.style.cssText="position:absolute;width:20px;height:"+h+"px;right:0px;border-radius: 8px;background-color: "+clr+";";document.getElementById(key).appendChild(right);foo("slideshow_right"+(c).toString(),key);c=c+1;}}</script>;'
        var part3='<!DOCTYPE html><html lang="en"><head><body style="height: 1000px;background:'+document.getElementById("prev").style.background+'; display:inline-block;">'
        var FileSaver = require('file-saver');
        if(Object.keys(this.state.data_ss).length!==0)
            var blob = new Blob([part1+part2+final], {type: "text/plain;charset=utf-8"});
        else
            var blob = new Blob([part3+final], {type: "text/plain;charset=utf-8"});
        FileSaver.saveAs(blob, document.getElementById("filename").value+".html");
        
  

  }
  
    render() {
        return (
            <div>
            <div id="prev" >
            {/* <a href="/Users/vishwas/Desktop/t2.png" download><h1>hi</h1></a> */}
            {/* <input type="text" placeholder="text"></input> */}
            </div>

            <div id="edit" >
            {/* <input name="example" type="text" id="example" value="Something" onfocus="value=''" /> */}
            {/* <input type="text" value="Default text" class="inactive" onFocus="toggleText(this);" onBlur="toggleText(this);"></input>
            <input name="example" type="text" id="example" size="50" value="EGTEXT" onfocus="if(this.value=='EGTEXT')this.value=''" onblur="if(this.value=='')this.value='EGTEXT'" />
            <input value="someStandardValue"/> */} 
            <h1 id="demo">Click on elements to edit</h1>
            
            </div>
            <div id="download">
            <button onClick={this.showAlert} >DOWNLOAD</button>
            <input id="filename" /> .html
            </div>
            
            </div>
        )
    }

}

export default Edit;





   // var e_obj = document.createElement('input');
        // e_obj.type = "file";
        // // e_obj.innerHTML="hiii"
        // e_obj.id = "eslide";
        // e_obj.setAttribute('multiple','');
        // document.getElementById("edit").appendChild(e_obj);
        // var a = document.createElement('a'); 
        // var link = document.createTextNode("This is link")
        // a.appendChild(link);
        // a.title = "This is Link";
        // a.href = "/segment0-1.png"
        // a.setAttribute('download','');
        // document.getElementById("edit").appendChild(a);
        // var e_done = document.createElement('button');
        // e_done.id = "e_done";
        // e_done.innerHTML="DONE"
        
        // e_obj.onchange = function(e) {
        //     var files = e.target.files;
        //     console.log(files[0])
            
        // }