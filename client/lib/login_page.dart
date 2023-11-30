import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LoginPage extends StatefulWidget {
  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();

  void _login(BuildContext context) async {
    if (_formKey.currentState!.validate()) {
      // Perform login verification (e.g., check credentials with a server).
      // For simplicity, we'll assume the login is successful.
      bool loginSuccessful = true;

      if (loginSuccessful) {
        SharedPreferences prefs = await SharedPreferences.getInstance();
        prefs.setBool('isLoggedIn', true);
        Navigator.pushReplacementNamed(context, '/dashboard');
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Invalid username or password')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background gradient
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [Colors.blue.shade700, Colors.blue.shade400, Colors.blue.shade100],
              ),
            ),
          ),
          Center(
            child: SingleChildScrollView(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Form(
                  key: _formKey,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Your logo here
                      CircleAvatar(
                        radius: 60,
                        backgroundColor: Colors.white, // Circle background color
                        child: Image.asset(
                          'assets/icon.png', // Replace this with your logo image
                          fit: BoxFit.fill, // Auto resize to fit the circle
                          width: 80,
                          height: 80,
                        ),
                      ),
                      SizedBox(height: 30),
                      // Username text field
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white, // Text box background color
                          borderRadius: BorderRadius.circular(8), // Rounded corners
                        ),
                        child: TextFormField(
                          controller: _usernameController,
                          decoration: InputDecoration(
                            prefixIcon: Icon(Icons.person),
                            labelText: 'Username',
                            border: InputBorder.none, // Remove the default border
                          ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter your username';
                            }
                            return null;
                          },
                        ),
                      ),
                      SizedBox(height: 16),
                      // Password text field
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white, // Text box background color
                          borderRadius: BorderRadius.circular(8), // Rounded corners
                        ),
                        child: TextFormField(
                          controller: _passwordController,
                          obscureText: true,
                          decoration: InputDecoration(
                            prefixIcon: Icon(Icons.lock),
                            labelText: 'Password',
                            border: InputBorder.none, // Remove the default border
                          ),
                          validator: (value) {
                            if (value == null || value.isEmpty) {
                              return 'Please enter your password';
                            }
                            return null;
                          },
                        ),
                      ),
                      SizedBox(height: 30),
                      ElevatedButton(
                        onPressed: () => _login(context), // Pass the context from here
                        style: ElevatedButton.styleFrom(
                          padding: EdgeInsets.symmetric(vertical: 16),
                        ),
                        child: Text('Login'),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
