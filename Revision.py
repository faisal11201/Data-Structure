#!/usr/bin/env python
# coding: utf-8

# ## Recursion

# Print n natural numbers

# In[ ]:


def print_n(n):
    if n==0:
        return
    print_n(n-1)
    print(n,end=' ')
n = int(input())
print_n(n)


# In[ ]:


def print_n2(n):
    if n==0:
        return
    print(n,end=' ')
    print_n2(n-1)
n = int(input())
print_n2(n)


# Nth fibonacci number

# In[ ]:


def fib(n):
    if n==1 or n==2:
        return 1
    fib_n_1 = fib(n-1)
    fib_n_2 = fib(n-2)
    return fib_n_1 + fib_n_2
n = int(input())
print(fib(n))


# Check List is Sorted Or Not

# In[ ]:


def isSorted(a):
    l = len(a)
    if l==0 or l==1:
        return True
    if a[0]>a[1]:
        return False
    smallerList = a[1:] # this line is very costly, copying the elemnts of a
    isSmallerSorted = isSorted(smallerList)
    return isSmallerSorted

a = [int(x) for x in input().split()]
print(isSorted(a))


# In[ ]:


def isSortedBetter(a,si):
    l=len(a)
    if si ==l-1 or si==l:
        return True
    if a[si]>a[si+1]:
        return False
    issmallerSorted = isSortedBetter(a,si+1)
    return issmallerSorted
a = [int(x) for x in input().split()]
print(isSortedBetter(a,0))


# First index of a number

# In[ ]:


def firstIndex(a,x):
    l = len(a)
    if l==0:
        return -1
    if a[0]==x:
        return 0
    smaller = a[1:]
    isFoundSmaller = firstIndex(smaller,x)
    if isFoundSmaller==-1:
        return -1
    else:
        return 1 + isFoundSmaller
    
a = [1,2,30,4,5,6,7,8,9,10]
x=10
print(firstIndex(a,x))


# In[ ]:


def firstIndexBetter(a,x,si):
    l = len(a)
    if si == l:
        return -1
    if a[si]==x:
        return si
    isFoundSmaller = firstIndex(a,x,si+1)
    return isFoundSmaller
a = [1,2,3,5,6,8,10]
x = 8
print(firstIndexBetter(a,x,0))


# Last Index of a number

# In[ ]:


def lastIndex(a,x):
    l=len(a)
    if l==0:
        return -1
    smallerList=a[1:]
    smallerListoutput = lastIndex(smallerList,x)
    if smallerListoutput!=-1:
        return smallerListoutput+1
    else:
        if a[0]==x:
            return 0
        else:
            return -1
        
a = [1,2,3,4,5,4,6,4,7,8]
print(lastIndex(a,4))


# In[ ]:


def lastIndexBetter(a,x,si):
    l=len(a)
    if si == l:
        return -1
    smallerListoutput = lastIndexBetter(a,x,si+1)
    if smallerListoutput!=-1:
        return smallerListoutput
    else:
        if a[si]==x:
            return si
        else:
            return -1
        
a = [1,2,3,4,5,4,6,4,7,8]
print(lastIndexBetter(a,4,0))            


# Replace char 

# In[ ]:


def replaceChar(s,a,b):
    if len(s)==0:
        return s
    smalleroutput = replaceChar(s[1:],a,b)
    if s[0]==a:
        return b + smalleroutput
    else:
        return s[0] + smalleroutput
    
print(replaceChar('daceebcc','c','x'))


# Remove x

# In[ ]:


def removeX(s):
    if len(s)==0:
        return s
    smalleroutput = removeX(s[1:])
    if s[0]=='x':
        return '' + smalleroutput
    else:
        return s[0] + smalleroutput
print(removeX('daxeebxx'))    


# Replace Pi with 3.14

# In[ ]:


def replacePI(s):
    if len(s)==0 or len(s)==1:
        return s
    if s[0]=='p' and s[1]=='i':
        smalloutput = replacePI(s[2:])
        return "3.14" + smalloutput
    else:
        smalloutput=replacePI(s[1:])
        return s[0] + smalloutput
    
print(replacePI("dappi"))    


# Remove consequtive duplicates

# In[ ]:


def removeDuplicates(s):
    if len(s)==0 or len(s)==1:
        return s
    if s[0]==s[1]:
        smalloutput = removeDuplicates(s[1:])
    else: 
        smalloutput = s[0] + removeDuplicates(s[1:])
    return smalloutput
s = input()
print(removeDuplicates(s))


# Binary Search using recursion

# In[ ]:


def binarySearch(a,x,si,ei):
    if si>ei:
        return -1
    mid = (si + ei)//2
    if a[mid]==x:
        return mid
    elif a[mid]>x:
        return binarySearch(a,x,si,mid-1)
    else:
        return binarySearch(a,x,mid+1,ei)
    
a=[int(x) for x in input().split()]
x = int(input())
si=0
ei = len(a)-1
print(binarySearch(a,x,si,ei))
        


# Merge sort using recursion

# In[ ]:


def merge(l1,l2,a):
    i=0
    j=0
    k=0
    while i<len(l1) and j<len(l2):
        if l1[i]<l2[j]:
            a[k]=l1[i]
            i=i+1
            k=k+1
        elif l1[i]>l2[j]:
            a[k]=l2[j]
            j=j+1
            k=k+1
    while i<len(l1):
        a[k]=l1[i]
        i=i+1
        k=k+1
    while j<len(l2):
        a[k]=l2[j]
        k=k+1
        j=j+1
    return a    
def mergeSort(a):
    if len(a)==0 or len(a)==1:
        return
    mid=len(a)//2
    l1=a[:mid]
    l2=a[mid:]
    mergeSort(l1)
    mergeSort(l2)
    return merge(l1,l2,a)
 
a=[4,3,2,14,1]
mergeSort(a)
for i in a:
    print(i)


# Quick Sort

# In[ ]:


def partition(a,si,ei):
    pivot = a[si]
    count = 0
    for i in range(si,ei+1):
        if a[i]<pivot:
            count+=1
    a[si+count],a[si] = a[si],a[si+count]
    pivot_index = si + count
    i=si
    j=ei
    while i<j:
        if a[i]<pivot:
            i=i+1
        elif a[j]>=pivot:
            j=j-1
        else:
            a[i],a[j] = a[j],a[i]
            i=i+1
            j=j-1
    return pivot_index        
def quickSort(a,si,ei):
    if si>=ei:
        return
    i = partition(a,si,ei)
    quickSort(a,si,i-1)
    quickSort(a,i+1,ei)
a = [int(x) for x in input().split()]
quickSort(a,0,len(a)-1)
for i in a:
    print(i,end=' ')


# Tower Of Hanoi

# In[ ]:


def towerofhanoi(n, source, aux, dest):
    if n==0:
        return
    if n==1:
        print(source,dest)
        return
    towerofhanoi(n-1, source, dest, aux)  #a,c,b
    print(source,dest)
    towerofhanoi(n-1,aux,source,dest)    #b,a,c
    return
   
n=int(input())
towerofhanoi(n, 'a', 'b', 'c')


# Geometric Sum
# 
# 1 + 1/2 + 1/4 + 1/8 + ... + 1/(2^k)

# In[ ]:


def geometricSum(k):
    if k==0:
        return 1
    sum = 1/(2**k)
    return sum + geometricSum(k-1)
k=int(input())
sum=(geometricSum(k))
print("{:.5f}".format(sum))


# Check Palindrome using recursion

# In[ ]:


def checkPalindrome(s,si,ei):
    if si>=ei:
        return True
    if s[si]==s[ei]:
        return checkPalindrome(s,si+1,ei-1)
    else:
        return False
s = input()
ei = len(s)-1
print(checkPalindrome(s,0,ei))


# Sum of digits

# In[ ]:


def sumOfdigits(n):
    if n==0:
        return 0
    rem = n%10
    return rem + sumOfdigits(n//10)
n = int(input())
print(sumOfdigits(n))


# Multiplication Recursive

# In[ ]:


def product(x,y):
#     if x<y:
#         return product(y,x)
    if y!=0:
        return x + product(x,y-1)
    
    else:
        return 0
	
x = int(input())
y = int(input())
print(product(x,y))


# COunt no of zeroes

# In[ ]:


def countZeroes(n):
    if n<0:
        n*=-1
    if n<10:
        if n==0:
            return 1
        else:
            return 0
    smallAns = countZeroes(n//10)
    if n%10==0:
        smallAns+=1
    return smallAns    
n = int(input())
print(countZeroes(n))
        
    


# ## LINKED LIST

# Taking Input( time complexity o(n2) )

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None        
def takeInput():
    inputList = [int(x) for x in input().split()]    
    head = None
    for currdata in inputList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head  = newNode
        else:
            curr = head
            while curr.next is not None:
                curr=curr.next
            curr.next = newNode
            
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None")    
head=takeInput()
printLL(head)
printLL(head)
                


# Taking input better approach ( O(n) )

# In[ ]:


class Node:
    
    def __init__(self,data):
        self.data=data
        self.next=None
        
def takeInputB():
    inputList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inputList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
head = takeInputB()
printLL(head)


# Length of Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inputList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inputList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next=newNode
            tail = newNode
    return head
def length(head):
    l=0
    while head is not None:
        l+=1
        head = head.next
    return l
head = takeInput()
l = length(head)
print(l)


# Print ith node

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inputList = [int(x) for x in input().split()]
    head = None
    tail =None
    for currdata in inputList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head ==None:
            head = newNode
            tail =newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def printIth(head,i):
    curr = 0
    start = head
    while curr!=i:
        start = start.next
        curr+=1
    print(start.data)
head = takeInput()
printLL(head)
i = int(input())
printIth(head,i)

        
        


# Insert At Ith postion

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def length(head):
    l=0
    while head is not None:
        l+=1
        head = head.next
    return l 

def insertAtIth(head,i,data):
    if i<0 or i>length(head):
        return head
    count=0
    prev = None
    curr = head
    while count<i:
        prev = curr
        curr=curr.next
        count+=1
        
    newNode = Node(data)
    if prev is not None:
        prev.next = newNode
    else:
        head = newNode
    newNode.next = curr
    return head                
def takeInput():
    inputList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inputList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head =newNode
            tail =newNode
        else:
            tail.next = newNode
            tail = newNode
    return head


head = takeInput()
printLL(head)
i = int(input())
data= int(input())
head = insertAtIth(head,i,data)
printLL(head)
        
    
    


# Delete Node

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def length(head):
    l=0
    while head is not None:
        l+=1
        head = head.next
    return l 
def delLL(head,i):
    if i==0:
        head = head.next
        return head
    if i<length(head):
        curr = head
        prev = None
        count=0
        while count<i:
            prev = curr
            curr =curr.next
            count+=1
        prev.next = curr.next
        curr.next = None
        return head
    else:
        return head
    
head = takeInput()
printLL(head)
i = int(input())
head = delLL(head,i)
printLL(head)

    


# Length of Linked List Recursive

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def LengthLLR(head):
    if head == None:
        return 0
    return 1 + LengthLLR(head.next)
head = takeInput()
print(LengthLLR(head))


# Insert At Ith position recursively

# In[2]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def insertATithR(head,i,data):
    if i<0:
        return head
    if i==0:
        newNode = Node(data)
        newNode.next = head
        return newNode
    if head is None:
        return None
    smallhead = insertATithR(head.next,i-1,data)
    head.next = smallhead
    return head
head = takeInput()
printLL(head)
i = int(input())
data = int(input())
head = insertATithR(head,i,data)
printLL(head)


# Delete Node Recursively

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def deleteR(head,i):
    if i==0:
        head = head.next
        return head
    if head is None:
        return None
    smallhead = deleteR(head.next,i-1)
    head.next = smallhead
    return head

head = takeInput()
printLL(head)
i = int(input())
head = deleteR(head,i)
printLL(head)
    


# Find a Node in LL

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def findNode(head,n):
    index = 0
    while head.data!=n:
        index+=1
        head = head.next
        if head is None:
            return -1
    return index  
head = takeInput()
printLL(head)
n = int(input())
print(findNode(head,n))


# Append Last N elements to front
# 
# I/P:1 2 3 4 5 -1
# 
# n = 3
# 
# O/P: 3 4 5 1 2 -1

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def length(head):
    l=0
    while head is not None:
        l+=1
        head = head.next
    return l 
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 

def appendLastN(head,n):
    count = length(head) - n
    i=0
    prev = None
    curr = head
    while i<count:
        prev = curr
        curr = curr.next
        i+=1
    newhead = curr
    prev.next = None
    curr = newhead
    temp = None
    while curr is not None:
        temp = curr
        curr=curr.next
    temp.next = head
    return newhead

head = takeInput()
printLL(head)
n = int(input())
head = appendLastN(head,n)
printLL(head)
        
        
    


# Eliminates  Consecutive Duplicates from Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def length(head):
    l=0
    while head is not None:
        l+=1
        head = head.next
    return l 
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 

def eliminatesDuplicates(head):
    if head is None:
        return head
    currhead = head
    while currhead.next is not None:
        if currhead.data==currhead.next.data:
            currhead.next = currhead.next.next
        else:
            currhead = currhead.next
    return head        

head = takeInput()
printLL(head)
head = eliminatesDuplicates(head)
printLL(head)


# Print Linked List in reverse order(just print)

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLLreverse(head):
    if head is None:
        return
    printLLreverse(head.next)
    print(head.data,end=' ')
head = takeInput()
printLLreverse(head)
    


# Palindrome Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def isPalindrome(head):
    while head is not None:
        l1.append(head.data)
        head=head.next
    return l1
def reverse(head):
    if head is None:
        return
    reverse(head.next)
    l2.append(head.data)
    return l2
head = takeInput()
l1 = []
l2=[]
isPalindrome(head)
reverse(head)
if l1==l2:
    print("true")
else:
    print('false')

    


# Reverse the Linked List(we have to change the connections)
# 

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
def takeInput():
    inpList = [int(s) for s in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)                           
        if head is None:
            head = newNode
            tail = newNode
            
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def reverseLL1(head):
    if head is None or head.next is None:
        return head
    smallhead = reverseLL1(head.next)
    curr = smallhead
    while curr.next is not None:
        curr = curr.next
    curr.next = head
    head.next = None
    return smallhead

head = takeInput()
printLL(head)
head = reverseLL1(head)
printLL(head)

# time complexity: t(n) = t(n-1) + n-1
#                t(n-1) = t(n-2) + n-2
#                 t(n-2)  = t(n-3) + n-3
#               ...
#             t(1) = 0
#             on solving we get t(n) = n^2
            
    
   


# Reverse the linked list in O(n)

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inpList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None") 
def reverseLL2(head):
    if head is None or head.next is None:
        return head,head
    smallhead,smalltail = reverseLL2(head.next)
    smalltail.next = head
    head.next =None
    return smallhead,head
head = takeInput()
printLL(head)
head,tail  = reverseLL2(head)
printLL(head)


# Reverse the Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inpList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None")
def reverseLL3(head):
    if head is None or head.next is None:
        return head
    smallhead = reverseLL3(head.next)
    tail = head.next
    tail.next = head
    head.next = None
    return smallhead
head = takeInput()
printLL(head)
head = reverseLL3(head)
printLL(head)


# Reverse Linked List Iteratively

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inpList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None")
def reverseLLI(head):
    curr = head
    prev = None
    Next = None
    while curr is not None:
        Next = curr.next
        curr.next = prev
        prev  = curr
        curr = Next
    head = prev
    return head

head = takeInput()
printLL(head)
head = reverseLLI(head)
printLL(head)


# Mid Point of Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inpList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None")
def midPoint(head):
    if head is None:
        return head
    slow = head
    fast =head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow
head = takeInput()
printLL(head)
midpoint = midPoint(head)
print(midpoint.data)


# Merge two sorted Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inpList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None")
def mergeLL(head1,head2):
    fhead = None
    ftail = None
    if head1.data<=head2.data:
        fhead = head1
        ftail = head1
        head1 = head1.next
    else:
        fhead = head2
        ftail = head2
        head2 = head2.next
    while head1 is not None and head2 is not None:
        if head1.data <=head2.data:
            ftail.next = head1
            ftail = head1
            head1 = head1.next
        else:
            ftail.next = head2
            ftail = head2
            head2 = head2.next
    if head1 is not None:
        ftail.next = head1
    else:
        ftail.next = head2
    return fhead    
head1 = takeInput()
head2 =takeInput()
fhead = mergeLL(head1,head2)
printLL(fhead)
        
        


# Merge sort in Linked List

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
def takeInput():
    inpList = [int(x) for x in input().split()]
    head = None
    tail = None
    for currdata in inpList:
        if currdata==-1:
            break
        newNode = Node(currdata)
        if head is None:
            head = newNode
            tail = newNode
        else:
            tail.next = newNode
            tail = newNode
    return head
def printLL(head):
    while head is not None:
        print(str(head.data) + "->",end='')
        head=head.next
    print("None")
def midPoint(head):
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow

def mergeLL(head1,head2):
    fhead = None
    ftail = None
    if head1.data<=head2.data:
        fhead = head1
        ftail = head1
        head1 = head1.next
    else:
        fhead = head2
        ftail = head2
        head2 = head2.next
    while head1 is not None and head2 is not None:
        if head1.data <=head2.data:
            ftail.next = head1
            ftail = head1
            head1 = head1.next
        else:
            ftail.next = head2
            ftail = head2
            head2 = head2.next
    if head1 is not None:
        ftail.next = head1
    else:
        ftail.next = head2
    return fhead    

def mergeSortLL(head):
    if head is None or head.next is None:
        return head
    mid = midPoint(head)
    head2 = mid.next
    mid.next = None
    head1 = mergeSortLL(head)
    head2 = mergeSortLL(head2)
    mergell =  mergeLL(head1,head2)
    return mergell
    
head = takeInput()
printLL(head)
head = mergeSortLL(head)
printLL(head)
    


# ## STACK

# In[ ]:


class Stack:
    def __init__(self):
        self.__data=[]
    def push(self,item):
        self.__data.append(item)
    def pop(self):
        if self.isEmpty():
            print("Hey! Stack is empty")
            return
        return self.__data.pop()
    
    def top(self):
        if self.isEmpty():
            print("Hey! Stack is empty")
            return
        return self.__data[len(self.__data)-1]
    def size(self):
        return len(self.__data)
    def isEmpty(self):
        return self.size()==0
s = Stack()
s.push(12)
s.push(13)
s.push(15)
s.push(20)
while s.isEmpty() is False:
    print(s.pop())
s.top()
    
        
        
        


# Stack using Linked list
# 

# In[ ]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
class Stack:
    def __init__(self):
        self.__head = None
        self.__count = 0
    def push(self,item):
        newNode = Node(item)
        newNode.next = self.__head
        self.__head = newNode
        self.__count+=1
    def pop(self):
        if self.isEmpty():
            print("Stack is empty")
            return
        element = self.__head.data
        self.__head = self.__head.next
        self.__count-=1
        return element
    def top(self):
        if self.isEmpty():
            print("Stack is empty")
            return
        return self.__head.data
    def size(self):
        return self.__count
    def isEmpty(self):
        return self.size()==0
s = Stack()
s.push(12)
s.push(13)
s.push(14)
while s.isEmpty() is False:
    print(s.pop())
(s.top())    
        
    


# Inbuilt Stack 

# In[ ]:


#inbuilt stack
s = [1,2,3]
s.append(4)
s.append(5)
print(s.pop())
print(s.pop())


# In[ ]:


# inbuilt stack
import queue
q = queue.LifoQueue()
q.put(1)
q.put(2)
q.put(3)
while not q.empty():
    print(q.get())


# Inbuilt queue

# In[ ]:


import queue
q = queue.Queue()
q.put(1)
q.put(2)
q.put(3)
q.put(4)
while not q.empty():
    print(q.get())


# Check Balanced parenthesis

# In[ ]:


def checkbalanced(string):
    s = []
    for char in string:
        if char in '({[':
            s.append(char)
        elif char is ')':
            if (not s or s[-1]!='('):
                return False
            s.pop()
        elif char is '}':
            if (not s or s[-1]!='{'):
                return False
            s.pop()
        elif char is ']':
            if (not s or s[-1]!='['):
                return False
            s.pop()
    if not s:
        return True
    else:
        return False
string = input()
print(checkbalanced(string))


# Reverse a stack using another stack(dont change the reference)

# In[ ]:


def reverseStack(s1,s2):
    if len(s1)<=1:
        return
    while len(s1)!=1:
        s2.append(s1.pop())
    ele = s1.pop()
    while len(s2)!=0:
        s1.append(s2.pop())
    reverseStack(s1,s2)    
    s1.append(ele)
s1 = [int(x) for x in input().split()]
s2= []
reverseStack(s1,s2)
while len(s1)!=0:
    print(s1.pop(),end=' ')
        


# Check Redundant Brackets

# In[ ]:


def checkRedundant(string):
    s=[]
    for char in string:
        if char != ')':
            s.append(char)
        else:
            if s[-1]=='(':
                return True
            while s[-1]!="(":
                s.pop()
            s.pop()
    return False        
string = input()
print(checkRedundant(string))
                


# Given a string expression which consists only ‘}’ and ‘{‘. The expression may not be balanced. You need to find the minimum number of bracket reversals which are required to make the expression balanced.
# Return -1 if the given expression can't be balanced.

# In[15]:


def minBracketReversal(string):
    if(len(string) == 0):
        return 0
    if(len(string)%2 != 0):
        return -1
    str = []
    for char in string:
        if char == '{':
            str.append(char)
        else:
            if(len(str) > 0 and str[-1] == '{'):
                str.pop()
            else:
                str.append(char)
    count = 0
    while len(str)!=0:
        c1 = str.pop()
        c2 = str.pop()
        if c1==c2:
            count+=1
        else:
            count+=2
    return count

string = input()
print(minBracketReversal(string))
    
            
                
    


# # QUEUES

# Queue using Array

# In[9]:


class queueUsingArray:
    def __init__(self):
        self.__arr = []
        self.__count=0
        self.__front = 0
    def enqueue(self,data):
        self.__arr.append(data)
        self.__count+=1
    def dequeue(self):
        if self.__count==0:
            return -1
        element = self.__arr[self.__front]
        self.__front+=1
        self.__count-=1
        
        return element
    def front(self):
        if self.__count==0:
            return -1
        return self.__arr[self.__front]
            
    def size(self):
        return self.__count
    def isEmpty(self):
        return self.size()==0
    
q = queueUsingArray()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
while q.isEmpty() is False:
    print(q.front())
    q.dequeue()
print(q.dequeue())    

    
        
        


# Queue Using LL

# In[8]:


class Node:
    def __init__(self,data):
        self.data=data
        self.next = None
class queueUsingLL:
    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__count= 0
    def enqueue(self,data):
        newNode = Node(data)
        if self.__head is None:
            self.__head = newNode
            self.__tail = newNode
        else:
            self.__tail.next = newNode
            self.__tail = newNode
        self.__count+=1
    def dequeue(self):
        if self.__head is None:
            return -1
        element = self.__head.data
        self.__head = self.__head.next
        self.__count=self.__count-1
        return element  
    def front(self):
        if self.__head==None:
            return -1
        element = self.__head.data       
        return element
    def size(self):
        return self.__count
    def isEmpty(self):
        return self.size()==0
q = queueUsingLL()
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
while q.isEmpty() is False:
    print(q.front())
    q.dequeue()
print(q.dequeue())    
            


# Queue using two stacks

# In[7]:


class queueUsingTwoStack:
    def __init__(self):
        self.__s1=[]
        self.__s2=[]
        self.__count=0
# enqueue in o(n)
    def enqueue(self,data):
        while len(self.__s1)!=0:
            self.__s2.append(self.__s1.pop())
        self.__s1.append(data)
        while len(self.__s2)!=0:
            self.__s1.append(self.__s2.pop())
        return
# dequeue in o(1)
    def dequeue(self):
        if len(self.__s1)==0:
            return -1
        return self.__s1.pop()
    def front(self):
        if len(self.__s1)==0:
            return -1
        return self.__s1[-1]
    def size(self):
        return len(self.__s1)
    def isEmpty(self):
        return self.size()==0
q =  queueUsingTwoStack()   
        
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
while q.isEmpty() is False:
    print(q.front())
    q.dequeue()
    
print(q.dequeue())        
    
            
        


# In[6]:


t = int(input())
while t!=0:
    n = int(input())
    chef = []
    morty = []
    while n!=0:
        l=[int(x) for x in input().split()]
        a = l[0]
        b = l[1]
       
        sa=0
        sb =0
        for i in str(a):
            sa+=int(i)
        for i in str(b):
            sb+=int(i)
        if sa>sb:
            chef.append(0)
        elif sa<sb:
            morty.append(1)
        n=n-1
    if len(chef)>len(morty):
        print(0,len(chef),end=' ')
    elif len(chef)<len(morty):
        print(1,len(morty),end=' ')
    else:
        print(2,len(chef),end=' ')
    t=t-1
    
        
    
            
            
    
    


# Reverse queue

# In[21]:


import queue
def reverseQueue(q):
    if q.empty() is True:
        return
    else:
        data = q.get()
        reverseQueue(q)
        q.put(data)
l = [int(x) for x in input().split()]        
q = queue.Queue()
for i in l:
    q.put(i)    
reverseQueue(q)
while q.empty() is False:
    print(q.get())


# Reverse first k elements of queue

# In[47]:


import queue
def reverseFirstK(Queue,k):
    if Queue.empty() is True or k>Queue.qsize():
        return
    if k<=0:
        return
    stack = []
    for i in range(k):
        stack.append(Queue.queue[0])
        Queue.get()
    while len(stack)!=0:
        Queue.put(stack[-1])
        stack.pop()
       
    for i in range(Queue.qsize()-k):
        Queue.put(Queue.queue[0])
        Queue.get()
n = int(input())      
li = [int(ele) for ele in input().split()]
q = queue.Queue()
for ele in li:
	q.put(ele)
k = int(input())
reverseFirstK(q,k) 
while q.empty() is False:
    print(q.get(),end=' ')
    n-=1
        


# ## Binary Tree

# In[2]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left = None
        self.right = None
def printTree(root):
    if root is None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end='')
    print()
    printTree(root.left)
    printTree(root.right)
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
root = treeInput()
printTree(root)


# Count Number of Nodes

# In[4]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left = None
        self.right = None
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end='')
    print()
    printTree(root.left)
    printTree(root.right)
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root

def numNodes(root):
    if root is None:
        return 0
    leftcount = numNodes(root.left)
    rightcount =numNodes(root.right)
    return 1 + leftcount+rightcount
root = treeInput()
printTree(root)
print("Number of Nodes: ",numNodes(root))


# Sum of all Nodes data

# In[8]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left = None
        self.right = None
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end='')
    print()
    printTree(root.left)
    printTree(root.right)
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def sumNodes(root):
    if root is None:
        return 0
    return root.data + sumNodes(root.left) + sumNodes(root.right)
root = treeInput()
printTree(root)
print(sumNodes(root))    


#  Pre Order Traversal in a Tree
# 
# root left right

# In[10]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left = None
        self.right = None
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end='')
    print()
    printTree(root.left)
    printTree(root.right)
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def preOrder(root):
    if root ==None:
        return
    print(root.data,end=' ')
    preOrder(root.left)
    preOrder(root.right)
root = treeInput()
printTree(root)
preOrder(root)


# Post Order Tree Traversal
# 
# left right root

# In[14]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left = None
        self.right = None
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end='')
    print()
    printTree(root.left)
    printTree(root.right)
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def postOrder(root):
    if root ==None:
        return
    postOrder(root.left)
    postOrder(root.right)
    print(root.data,end=' ')

root = treeInput()    
printTree(root)
postOrder(root)


# In[2]:


t = int(input())
while t!=0:
    n = int(input())
    a = [int(x) for x in input().split()]
    sum=0
    i=0
    while i<n-1:
        sum+=abs(a[i]-a[i+1])
        i+=1
    ans = sum-n+1
    print(ans)
    t=t-1
    


# Largest Data of a Tree

# In[3]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print(root.left.data,end=',')
    if root.right!=None:
        print(root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
    
def largestData(root):
    if root==None:
        return -1 # idealy return -inf
    leftLargest = largestData(root.left)
    rightLargest = largestData(root.right)
    largest = max(root.data,leftLargest,rightLargest)
    return largest
root = treeInput()
printTree(root)
print(largestData(root))
    


# Given a Binary Tree and an integer x, find and return the count of nodes which are having data greater than x.

# In[6]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print(root.left.data,end=',')
    if root.right!=None:
        print(root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
    
def countNodes(root,x):
    if root==None:
        return 0
    if root.data>x:
        return 1 + countNodes(root.left,x) + countNodes(root.right,x)
    else:
        return countNodes(root.left,x) + countNodes(root.right,x)
root = treeInput()
printTree(root)
x = int(input())
print(countNodes(root,x))


# Height of a Tree

# In[12]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def height(root):
    if root==None:
        return 0
    leftHeight = height(root.left)
    rightHeight = height(root.right)
    return 1 + max(leftHeight,rightHeight)
root = treeInput()
printTree(root)
print(height(root))


# Count Number of leaf Nodes

# In[13]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def numLeaf(root):
    if root==None:
        return 0
    if root.left==None and root.right==None:
        return 1
    numLeafLeft = numLeaf(root.left)
    numLeafRight = numLeaf(root.right)
    return numLeafLeft + numLeafRight
root = treeInput()
printTree(root)
print(numLeaf(root))


# Print all nodes at depth k

# In[15]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)

def printAtK(root,k):
    if root==None:
        return
    if k==0:
        print(root.data,end=' ')
        return
    printAtK(root.left,k-1)
    printAtK(root.right,k-1)
root = treeInput()
printTree(root)
k = int(input())
(printAtK(root,k))
    


# Replace Node with depth

# In[17]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def replaceNodeWithDepth(root,d):
    if root==None:
        return
    root.data = d
    replaceNodeWithDepth(root.left,d+1)
    replaceNodeWithDepth(root.right,d+1)
    return root
root = treeInput()
printTree(root)
root = replaceNodeWithDepth(root,0)
printTree(root)


# Is Node Present

# In[23]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def isNodePresent(root,x):
    if root==None:
        return False
    if root.data==x:
        return True
    left = isNodePresent(root.left,x)
    right = isNodePresent(root.right,x)
    if left:
        return True
    if right:
        return True
root = treeInput()
printTree(root)
x = int(input())
present = (isNodePresent(root,x))
if present:
    print('true')
else:
    print('false')


# Remove Leaf Nodes

# In[2]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
    
def removeLeaf(root):
    if root is None:
        return
    if root.left ==None and root.right==None:
        return None
    root.left = removeLeaf(root.left)
    root.right = removeLeaf(root.right)
    return root
root = treeInput()
printTree(root)
root = removeLeaf(root)
printTree(root)


# Mirror image of given tree

# In[ ]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def mirror(root):
    if root is None:
        return 
    root.left = mirror(root.left)
    root.right = mirror(root.right)
    root.left,root.right = root.right,root.left
    return root
root = treeInput()
printTree(root)
root = mirror(root)
printTree(root)


# Check is Tree Balanced?(time complexity = o(nlogn))

# In[10]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def height(root):
    if root==None:
        return 0
    return 1 + max(height(root.left),height(root.right))
def isBalanced(root):
    if root is None:
        return True
    lh = height(root.left)
    rh = height(root.right)
    if lh-rh>1 or rh-lh>1:
        return False
    isLeftBalanced = isBalanced(root.left)
    isRightBalanced = isBalanced(root.right)
    if isLeftBalanced and isRightBalanced:
        return True
    return False
root = treeInput()
printTree(root)
print(isBalanced(root))


# Check Balanced better solution (o(n))

# In[2]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def getheightAndcheckBalanced(root):
    if root==None:
        return 0,True
    lh,isLeftBalanced = getheightAndcheckBalanced(root.left)
    rh,isRightBalanced = getheightAndcheckBalanced(root.right)
    h = 1 + max(lh,rh)
    if lh-rh>1 or rh-lh>1:
        return h,False
    if isLeftBalanced and isRightBalanced:
        return h,True
    else:
        return h,False
    
root = treeInput()
printTree(root)
print(getheightAndcheckBalanced(root))
    
    


# Diameter of a Binary Tree (Distance between two farthest node)

# In[5]:


class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def treeInput():
    rootData = int(input())
    if rootData==-1:
        return None
    root = BinaryTreeNode(rootData)
    leftTree = treeInput()
    rightTree = treeInput()
    root.left = leftTree
    root.right = rightTree
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
def height(root):
    if root==None:
        return 0
    lh = height(root.left)
    rh = height(root.right)
    return 1 + max(lh,rh)
def diameter(root):
    if root==None:
        return 0
    opt1 = height(root.left) + height(root.right)
    opt2 = diameter(root.left)
    opt3 = diameter(root.right)
    return max(opt1,opt2,opt3)
root = treeInput()
printTree(root)
print(diameter(root))


# Level wise input of tree
# 
# Algo:Take root input
# 
# add it to queue
# 
# while q is not empty{
# 
# take front of q->a
# 
# ask for its children
# 
# if valid children attach them to a
# 
# add them to queue
# 
# }

# In[5]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseTreeInput():
    q = queue.Queue()
    print("Enter root")
    rootData = int(input())
    if rootData == -1:
        return None
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("Enter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)    
    return root        
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    if root.left!=None:
        print("L",root.left.data,end=',')
    if root.right!=None:
        print("R",root.right.data,end=' ')
    print()
    printTree(root.left)
    printTree(root.right)
root = takeLevelWiseTreeInput()
printTree(root)


# Print LevelWise
# 
# Add root to queue
# 
# while q is not empty{
#     
#     take out first node->a
#     
#     print for a
#     check for a's children{
#     
#     if they are not none add them to queue
# }
# }
# }

# In[8]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseTreeInput():
    q = queue.Queue()
    print("Enter root")
    rootData = int(input())
    if rootData == -1:
        return None
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("Enter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)    
    return root

def printTreeLevelWise(root):
    if root == None:
        return None
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftchild = a.left
        if leftchild!=None:
            print("L",leftchild.data,end=',')
            q.put(leftchild)
        rightchild = a.right
        if rightchild!=None:
            print("R",rightchild.data,end=' ')
            q.put(rightchild)
        print()    
root = takeLevelWiseTreeInput()
printTreeLevelWise(root)


# Construct Tree Using preorder and Inorder

# In[6]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return None
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()    
def constructFromPreIn(pre,inorder):
    if len(pre)==0:
        return None
    rootData = pre[0]
    root = BinaryTreeNode(rootData)
    rootIndexInorder = -1
    for i in range(0,len(inorder)):
        if inorder[i]==rootData:
            rootIndexInorder = i
            break
    if rootIndexInorder == -1:
        return None
    leftInorder = inorder[0:rootIndexInorder]
    rightInorder = inorder[rootIndexInorder+1:]
    
    lenLeftSubtree = len(leftInorder)
    
    leftPreOrder = pre[1:lenLeftSubtree+1]
    rightPreOrder = pre[lenLeftSubtree+1:]
    
    leftChild = constructFromPreIn(leftPreOrder,leftInorder)
    rightChild = constructFromPreIn(rightPreOrder,rightInorder)
    
    root.left = leftChild
    root.right = rightChild
    
    return root
pre = [1,2,4,5,3,6,7]
inorder = [4,2,5,1,6,3,7]
root = constructFromPreIn(pre,inorder)
printTreeLevelWise(root)


        


# COnstruct Tree Using Postorder and Inorder

# In[9]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return None
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def buildTreePostOrder(postorder, inorder):
    if len(postorder)==0:
        return None
    rootData = postorder[len(postorder)-1]
    root = BinaryTreeNode(rootData)
    rootindexinorder=-1
    for i in range(0,len(inorder)):
        if inorder[i]==rootData:
            rootindexinorder=i
            break
    if rootindexinorder==-1:
        return None
    leftinorder = inorder[0:rootindexinorder]
    rightinorder = inorder[rootindexinorder+1:]
    
    lenleftsubtree=len(leftinorder)
    
    leftpostorder=postorder[0:lenleftsubtree]
    rightpostorder=postorder[lenleftsubtree:len(postorder)-1]
    
    leftchild = buildTreePostOrder(leftpostorder,leftinorder)
    rightchild= buildTreePostOrder(rightpostorder,rightinorder)
    
    root.left = leftchild
    root.right = rightchild
    return root

postorder =[8,4,5,2,6,7,3,1]
inorder =[4,8,2,5,1,6,3,7]
root = buildTreePostOrder(postorder,inorder)
printTreeLevelWise(root)


# Node to root path 

# In[1]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return None
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def nodeToRootPath(root,s):
    if root==None:
        return None
    if root.data==s:
        l = []
        l.append(root.data)
        return l
    leftOutput = nodeToRootPath(root.left,s)
    if leftOutput!=None:
        leftOutput.append(root.data)
        return leftOutput
    rightOutput = nodeToRootPath(root.right,s)
    if rightOutput!=None:
        rightOutput.append(root.data)
        return rightOutput
    else:
        return None
root = takeLevelWiseInput()    
printTreeLevelWise(root)
l = nodeToRootPath(root,5)
print(l)


# ## BST
# 
# left<root<right

# Search Node In BST

# In[15]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def searchInBST(root,x):
    if root==None:
        return False
    if root.data==x:
        return True
    if x<root.data:
        left = searchInBST(root.left,x)
        return left
    if x>=root.data:
        right = searchInBST(root.right,x)
        return right
    if left:
        return True
    if right:
        return True
root = takeLevelWiseInput()
x= int(input())
print(searchInBST(root,x))


# Print Elements in range k1 and k2

# In[2]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return 
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def printElementsInk1k2(root,k1,k2):
    if root == None:
        return None
    if root.data<k1:
        printElementsInk1k2(root.right,k1,k2)
    elif root.data>=k2:
        printElementsInk1k2(root.left,k1,k2)
    else:
        printElementsInk1k2(root.left,k1,k2)
        print(root.data,end=' ')
        printElementsInk1k2(root.right,k1,k2)
root = takeLevelWiseInput()
printTreeLevelWise(root)
k1 = int(input())
k2 = int(input())
printElementsInk1k2(root,k1,k2)


# Construct BST using a sorted array

# In[6]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def buildBST(arr):
    if len(arr)==0:
        return None
    mid = (len(arr))//2
    rootData= arr[mid]
    root = BinaryTreeNode(rootData)
    
    leftSubtree = arr[:mid]
    rightSubtree = arr[mid+1:]
    leftchild = buildBST(leftSubtree)
    rightchild = buildBST(rightSubtree)
    
    root.left = leftchild
    root.right = rightchild
    
    return root
def printTreeLevelWise(root):
    if root==None:
        return 
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
arr = [1,2,3,4,5,6,7,8]
root = buildBST(arr)
printTreeLevelWise(root)


# Check Is BST?

# In[1]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return 
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def minTree(root):
    if root is None:
        return 9999999
    leftMin = minTree(root.left)
    rightMin = minTree(root.right)
    return min(leftMin,rightMin,root.data)
def maxTree(root):
    if root is None:
        return -9999999
    leftMax = maxTree(root.left)
    rightMax = maxTree(root.right)
    return max(leftMax,rightMax,root.data)
def isBST(root):
    if root is None:
        return True
    leftMax = maxTree(root.left)
    rightMin = minTree(root.right)
    if root.data>rightMin or root.data<=left Max:
        return False
    isLeftBST = isBST(root.left)
    isRightBST = isBST(root.right)
    return isLeftBST and isRightBST
root = takeLevelWiseInput()
printTreeLevelWise(root)
print(isBST(root))
        


# Better solution for check Is BST??

# In[5]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return 
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def isBST2(root):
    if root is None:
        return 1000000,-1000000,True
    leftMin,leftMax,isLeftBST = isBST2(root.left)
    rightMin,rightMax,isRightBST = isBST2(root.right)
    minimum = min(leftMin,rightMin,root.data)
    maximum = max(leftMax,rightMax,root.data)
    isTreeBST =True
    if root.data<=leftMax or root.data>rightMin:
        isTreeBST = False
    if not(isLeftBST) or not(isRightBST):
        isTreeBST = False
    return minimum,maximum,isTreeBST
root = takeLevelWiseInput()
printTreeLevelWise(root)
print(isBST2(root))
        


# Another solution for checking BST

# In[7]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return 
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def isBST3(root,min_range,max_range):
    if root is None:
        return True
    if root.data<min_range or root.data > max_range:
        return False
    isLeftwithinRange = isBST3(root.left,min_range,root.data-1)
    isRightwithinRange = isBST3(root.right,root.data,max_range)
    return isLeftwithinRange and isRightwithinRange
root = takeLevelWiseInput()
printTreeLevelWise(root)
print(isBST3(root,-10000,10000))


# Find node to root path in BST

# In[5]:


import queue
class BinaryTreeNode:
    def __init__(self,data):
        self.data=data
        self.left=None
        self.right = None
def takeLevelWiseInput():
    print("Enter root")
    rootData = int(input())
    if rootData==-1:
        return None
    q = queue.Queue()
    root = BinaryTreeNode(rootData)
    q.put(root)
    while q.empty() is False:
        current_node = q.get()
        print("Enter left child of ", current_node.data)
        leftChildData = int(input())
        if leftChildData!=-1:
            leftChild = BinaryTreeNode(leftChildData)
            current_node.left = leftChild
            q.put(leftChild)
        print("ENter right child of ", current_node.data)
        rightChildData = int(input())
        if rightChildData!=-1:
            rightChild = BinaryTreeNode(rightChildData)
            current_node.right = rightChild
            q.put(rightChild)
    return root
def printTreeLevelWise(root):
    if root==None:
        return None
    q = queue.Queue()
    q.put(root)
    while q.empty() is False:
        a = q.get()
        print(a.data,end=':')
        leftChild = a.left
        if leftChild!=None:
            print("L",leftChild.data,end=",")
            q.put(leftChild)
        rightChild = a.right
        if rightChild!=None:
            print("R",rightChild.data,end=' ')
            q.put(rightChild)
        print()
def findPathBST(root,s):
    if root==None:
        return None
    if root.data==s:
        l=[]
        l.append(root.data)
        return l
    if s<root.data:
        Output = nodeToRootPath(root.left,s)
        if Output!=None:
            Output.append(root.data)
        return Output
    else:
        Output = nodeToRootPath(root.right,s)
        if Output!=None:
            Output.append(root.data)
        return Output
root = takeLevelWiseInput()    
printTreeLevelWise(root)
s = int(input())
l = nodeToRootPath(root,s)
print(l)
        


# ## GENERIC TREE

# In[2]:


class TreeNode:
    def __init__(self,data):
        self.data=data
        self.children = []
def takeInput():
    print("Enter root data")
    rootData = int(input())
    if rootData==-1:
        return None
    root = TreeNode(rootData)
    print("Enter number of children for ", rootData)
    childrenCount = int(input())
    for child in range(childrenCount):
        child = takeInput()
        root.children.append(child)
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    for child in root.children:
        print(child.data,end=",")
        
    print()    
    for child in root.children:
        printTree(child)
        
root = takeInput()        
printTree(root)


# Number of Nodes

# In[5]:


class TreeNode:
    def __init__(self,data):
        self.data=data
        self.children = []
def takeInput():
    print("Enter root data")
    rootData = int(input())
    if rootData==-1:
        return None
    root = TreeNode(rootData)
    print("Enter number of children for ", rootData)
    childrenCount = int(input())
    for child in range(childrenCount):
        child = takeInput()
        root.children.append(child)
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    for child in root.children:
        print(child.data,end=",")
        
    print()    
    for child in root.children:
        printTree(child)
        
def numNodes(root):
    if root==None: # not a base case,it is an edge case for empty tree
        return 0 
    count = 1
    for child in root.children:
        count = count + numNodes(child)
    return count    
root = takeInput()    
printTree(root)
print(numNodes(root))


# Node with maximum data

# In[3]:


class TreeNode:
    def __init__(self,data):
        self.data=data
        self.children = []
def takeInput():
    print("Enter root data")
    rootData = int(input())
    if rootData==-1:
        return None
    root = TreeNode(rootData)
    print("Enter number of children for ", rootData)
    childrenCount = int(input())
    for child in range(childrenCount):
        child = takeInput()
        root.children.append(child)
    return root
def printTree(root):
    if root==None:
        return
    print(root.data,end=':')
    for child in root.children:
        print(child.data,end=",")
        
    print()    
    for child in root.children:
        printTree(child)
        
def maxData(root):
    if root is None:
        return -1000000
    max = root.data
    for child in root.children:
        if max<maxData(child):
            max = maxData(child)
    return max
root = takeInput()
printTree(root)
print(maxData(root))
            


# ## Dictionaries/Maps

# Print all words with frequency k

# In[4]:


def printWordsFreqK(s,k):
    words = s.split()
    d={}
    for w in words:
        d[w]=d.get(w,0) + 1
    for w in d:
        if d[w] == k:
            print(w,end=' ')
s = input() 
k = int(input())
printWordsFreqK(s,k)


# Maximum Frequency

# In[7]:


def maxFreq(s):
    words = s.split()
    d = {}
    for w in words:
        d[w]=d.get(w,0) + 1
    maximum = -1
    for w in d:
        if maximum<d[w]:
            maximum = d[w]
            num = w
    print(num)   
s = input()
maxFreq(s)


# HashMap

# In[4]:


class MapNode:
    def __init__(self,key,value):
        self.key = key
        self.value = value
        self.next = None
class Map:
    def __init__(self):
        self.bucketSize = 5
        self.buckets = [None for i in range(self.bucketSize)]
        self.count = 0
    def size(self):
        return self.count

    def getBucketIndex(self,hc):
        return (abs(hc)%(self.bucketSize))
    
    def getValue(self,key):
        hc = hash(key)
        index = self.getBucketIndex(hc)
        head = self.buckets[index]
        
        while head is not None:
            if head.key == key:
                return head.value
            head = head.next
        return None
    def remove(self,key):
        hc = hash(key)
        index = self.getBucketIndex(hc)
        head = self.buckets[index]
        prev = None
        while head is not None:
            if head.key==key:
                if prev is None:
                    self.buckets[index] = head.next
                else:       
                    prev.next=head.next
                self.count-=1    
                return head.value
            prev = head
            head = head.next
    
        return None    
    def rehash(self):
        temp = self.buckets
        self.buckets = [None for i in range(2*self.bucketSize)]
        self.bucketSize = 2*self.bucketSize
        self.count = 0
        for head in temp:
            while head is not None:
                self.insert(head.key,head.value)
                head = head.next
    def loadFactor(self):
        return self.count/self.bucketSize
                
    def insert(self,key,value):
        
        hc = hash(key)
        index = self.getBucketIndex(hc)
        head = self.buckets[index]
        while head is not None:
            if head.key == key:
                head.value = value
                return
            head = head.next
        head = self.buckets[index]    
        newNode = MapNode(key,value)
        newNode.next = head
        self.buckets[index] = newNode
        self.count+=1
        loadFactor = self.count/self.bucketSize
        if loadFactor>=0.7:
            self.rehash()
m = Map()
for i in range(10):
    m.insert('abc' + str(i),i+1)
    print(m.loadFactor())
for i in range(10):
    print('abc' + str(i) + ':',m.getValue('abc' + str(i)))
    
# m.insert('Parikh', 2)
# print(m.size())
# m.insert('Rohan',7)
# print(m.size())
# m.insert('Parikh',9)
# print(m.size())
# print(m.getValue('Rohan'))
# print(m.getValue('Parikh'))
# m.remove('Rohan')
# print(m.getValue('Rohan'))        


# ## PRIORITY QUEUE

# Implementation of Priority Queue(MIN)

# In[1]:


class priorityQueueNode:
    def __init__(self,value,priority):
        self.value = value
        self.priority = priority
        
class priorityQueue:
    def __init__(self):
        self.pq = []
       
    def getSize(self):
        return len(self.pq)
    
    def isEmpty(self):
        return self.getSize()==0
    
    def getMin(self):
        if self.ismpty() is True:
            return None
        return self.pq[0].value
    
    def __percolateUp(self):
        childIndex = self.getSize()-1
        while childIndex>0:
            parentIndex = (childIndex-1)//2
            if self.pq[childIndex].priority<self.pq[parentIndex].priority:
                self.pq[childIndex],self.pq[parentIndex] = self.pq[parentIndex],self.pq[childIndex]
                childIndex = parentIndex
               
            else:
                break
    
    def insert(self,value,priority):
        pqNode = priorityQueueNode(value,priority)
        self.pq.append(pqNode)
        self.__percolateUp()
        
    def __percolateDown(self):
        parentIndex = 0
        leftChildIndex = 2*parentIndex + 1
        rightChildIndex = 2*parentIndex+2
        
        while leftChildIndex<self.getSize():
            minIndex = parentIndex
            if self.pq[minIndex].priority > self.pq[leftChildIndex].priority:
                minIndex = leftChildIndex
            if rightChildIndex < self.getSize() and self.pq[minIndex].priority > self.pq[rightChildIndex].priority:
                minIndex = rightChildIndex
                
            if minIndex==parentIndex:
                break
            else:
                self.pq[parentIndex],self.pq[minIndex] = self.pq[minIndex],self.pq[parentIndex]
                parentIndex = minIndex
                leftChildIndex = 2*parentIndex + 1
                rightChildIndex = 2*parentIndex+2 
                
        
        
    def removeMin(self):
        if self.isEmpty():
            return None
        ele = self.pq[0].value
        self.pq[0] = self.pq[self.getSize()-1]
        self.pq.pop()
        self.__percolateDown()
        return ele
    
pq = priorityQueue()
pq.insert('A',10)
pq.insert('C',5)
pq.insert('B',19) 
pq.insert('D',4)
for i in range(4):
    print(pq.removeMin())

        
    
    


# Heap Sort

# In[1]:


def down_heapify(arr,i,n):
    parentIndex = i
    leftChildIndex = 2*parentIndex +1
    rightChildIndex = 2*parentIndex +2
    
    while leftChildIndex<n:
        minIndex = parentIndex
        if arr[minIndex]>arr[leftChildIndex]:
            minIndex = leftChildIndex
        if rightChildIndex < n and arr[minIndex]  > arr[rightChildIndex]:
            minIndex = rightChildIndex
        if minIndex==parentIndex:
            return
        else:
            arr[minIndex],arr[parentIndex] = arr[parentIndex],arr[minIndex]
            parentIndex=minIndex
            leftChildIndex = 2*parentIndex +1
            rightChildIndex = 2*parentIndex +2
            
    return       
            
            
    
def heapSort(arr):
    # build the heap
    n = len(arr)
    for i in range(n//2-1,-1,-1):
        down_heapify(arr,i,n)
        
    #removing n elements and put them at correct position
    for i in range(n-1,0,-1):
        arr[0],arr[i] = arr[i],arr[0]
        down_heapify(arr,0,i)
    return arr    
        
arr = [int(x) for x in input().split()]
heapSort(arr)
for ele in arr:
    print(ele,end=' ')


# Inbuilt min heap

# In[6]:


import heapq
li=[1,5,4,8,7,9,11]
heapq.heapify(li)
print(li)


# In[7]:


heapq.heappush(li,2)
print(li)


# In[8]:


print(heapq.heappop(li))


# In[9]:


print(li)


# In[10]:


heapq.heapreplace(li,6)   #replace min element
print(li)


# Inbuilt max heap

# In[11]:


import heapq
li=[1,5,4,7,8,9,2,3]
heapq._heapify_max(li)
print(li)


# In[12]:


print(heapq._heappop_max(li))


# In[13]:


print(li)


# In[14]:


heapq._heapreplace_max(li,0)
print(li)


# In[15]:


li.append(6)
heapq._siftdown_max(li,0,len(li)-1)
print(li)


# K smallest element

# In[3]:


import heapq
def kSmallest(arr,k):
    heap = arr[:k]
    heapq._heapify_max(heap)
    n = len(arr)
    for i in range(k,n):
        if heap[0]>arr[i]:
            heapq._heapreplace_max(heap,arr[i])
    return heap        
arr = [int(x) for x in input().split()]
k = int(input())
elements = kSmallest(arr,k)
for ele in elements[::]:
    print(ele,end=' ')


# k Largest elements

# In[3]:


import heapq
def kLargest(arr,k):
    heap = arr[:k]
    heapq.heapify(heap)
    n = len(arr)
    for i in range(k,n):
        if heap[0]<arr[i]:
            heapq.heapreplace(heap,arr[i])
    return heap        
arr = [int(x) for x in input().split()]
k = int(input())
elements = kLargest(arr,k)
for ele in elements:
    print(ele,end=' ')


# kth largest element

# In[5]:


import heapq
def kthLargest(arr,k):
    heap = arr[:k]
    heapq.heapify(heap)
    n = len(arr)
    for i in range(k,n):
        if heap[0]<arr[i]:
            heapq.heapreplace(heap,arr[i])
    return heap[0]
arr = [int(x) for x in input().split()]
k = int(input())
print(kthLargest(arr,k))


# ## GRAPH

# Implementation of Graph (DFS and BFS)

# In[13]:


import queue
class Graph:
    def __init__(self,nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(nVertices)] for j in range(nVertices)]
    def addEdge(self,v1,v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
    def __dfsHelper(self,sv,visited):
        print(sv)
        visited[sv] = True
        for i in range(self.nVertices):
            if self.adjMatrix[sv][i]>0 and visited[i] is False:
                self.__dfsHelper(i,visited)
        
    def dfs(self):
        visited = [False for i in range(self.nVertices)]
        for i in range(self.nVertices):
            if visited[i] == False:
                self.__dfsHelper(i,visited)
        
    def __bfsHelper(self,sv,visited):
        q = queue.Queue()
        q.put(sv)
        visited[sv] = True
        
        while q.empty() is False:
            u = q.get()
            print(u)
            for i in range(self.nVertices):
                if self.adjMatrix[u][i] > 0 and visited[i] is False:
                    q.put(i)
                    visited[i] = True
    def bfs(self):
        visited = [False for i in range(self.nVertices)]
        for i in range(self.nVertices):
            if visited[i]==False:
                self.__bfsHelper(i,visited)
    def removeEdge(self,v1,v2):
        if self.containsEdge() is False:
            return
        self.adjMatrix[v1][v2]=0
        self.adjMatrix[v2][v1]=0
    def containsEdge(self,v1,v2):
        return True if self.adjMatrix[v1][v2]>0 else False
    
    def __str__(self):
        return str(self.adjMatrix)

g = Graph(5)
g.addEdge(0,1)
g.addEdge(0,2)
g.addEdge(1,3)
g.addEdge(2,3)
g.addEdge(2,4)
print("DFS Order:")
g.dfs()
print("BFS Order")
g.bfs()


# Has Path

# In[10]:


class Graph:
    def __init__(self,nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
        
    def addEdge(self,v1,v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
        
    def hasPathHelper(self,v1,v2,visited):
        if self.adjMatrix[v1][v2]>0:
            return True
        visited[v1]=True
        for i in range(self.nVertices):
            if self.adjMatrix[v1][i]>0 and visited[i] == False:
                if self.hasPathHelper(i,v2,visited) is True:
                    return True
        return False        
    def hasPath(self,v1,v2):
        visited = [False for i in range(self.nVertices)]
        return self.hasPathHelper(v1,v2,visited)
g = Graph(5)
g.addEdge(0,1)
g.addEdge(1,2)
g.addEdge(3,4)
print(g.hasPath(0,4))

        


# GET DFS PATH From v1 to v2

# In[23]:


class Graph:
    def __init__(self,nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
    
    def addEdge(self,v1,v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
     
    def __getPathHelper(self, v1, v2, visited, list):
        if v1==v2:
            list.append(v1)
            return list
        
        visited[v1]=True
        for i in range(self.nVertices):
            if self.adjMatrix[v1][i]>0 and visited[i] is False:
                li = self.__getPathHelper(i,v2,visited,list)
                if li is not None:
                    li.append(v1)
                    return li
                
        return None        
      
    def getDFSPath(self,v1,v2):
        visited = [False for i in range(self.nVertices)]
        return self.__getPathHelper(v1,v2,visited,[])
        
g = Graph(7)
g.addEdge(0,1)
g.addEdge(0,2)
g.addEdge(0,3)
g.addEdge(1,4)
g.addEdge(2,5)
g.addEdge(5,6)
li = g.getDFSPath(0,6)
for i in li:
    print(i)


# GET BFS PATH(BFS is the shortest path)

# In[27]:


class Graph:
    def __init__(self,nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
    
    def addEdge(self,v1,v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
     
    def __getBFSPathHelper(self,sv,ev,visited):
        mapp = {}
        q = queue.Queue()
        if self.adjMatrix[sv][ev]==1 and sv==ev:
            ans = []
            ans.append(sv)
            return ans
        
        q.put(sv)
        visited[sv] = True
        while q.empty() is False:
            front = q.get()
            for i in range(self.nVertices):
                if self.adjMatrix[front][i]==1 and visited[i] is False:
                    mapp[i]=front
                    q.put(i)
                    visited[i] = True
                    
                    if i==ev:
                        ans=[]
                        ans.append(ev)
                        value= mapp[ev]
                        
                        while value!=sv:
                            ans.append(value)
                            value = mapp[value]
                            
                        ans.append(value)
                        return ans
        return None        

                        
    def getBFSPath(self,sv,ev):
        visited = [False for i in range(self.nVertices)]
        return self.__getBFSPathHelper(sv,ev,visited)
    
g = Graph(7)
g.addEdge(0,1)
g.addEdge(0,2)
g.addEdge(0,3)
g.addEdge(1,4)
g.addEdge(2,5)
g.addEdge(5,6)
g.addEdge(3,6)
li = g.getBFSPath(0,6)
for i in li:
    print(i)
    
    
    


# IS GRAPH CONNECTED?

# In[3]:


class Graph:
    def __init__(self,nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
    
    def addEdge(self,v1,v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
        
    def __dfsHelper(self,sv,visited):
        visited[sv] = True
        for i in range(self.nVertices):
            if self.adjMatrix[sv][i]> 0 and visited[i] is False:
                self.__dfsHelper(i,visited)
        return visited        
    def dfs(self):
        visited = [False for i in range(self.nVertices)]
        return self.__dfsHelper(0,visited)

g = Graph(5)
g.addEdge(0,1)
g.addEdge(0,2)
g.addEdge(3,4)

visited = g.dfs()
for i in visited:
    if i is False:
        print('false')
        break
else:
    print('true')
        


# ALL CONNNECTED COMPONENT?

# In[2]:


class Graph:
    def __init__(self,nVertices):
        self.nVertices = nVertices
        self.adjMatrix = [[0 for i in range(self.nVertices)] for j in range(self.nVertices)]
    
    def addEdge(self,v1,v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
        
    def __dfsHelper(self,sv,visited,st):
        st.append(sv)
        visited[sv] = True
        for i in range(self.nVertices):
            if self.adjMatrix[sv][i]>0 and visited[i] is False:
                self.__dfsHelper(i,visited,st)
        return st              
      
    
    def connectedComponent(self):
        visited = [False for i in range(self.nVertices)]
        finalList = []
        for i in range(self.nVertices):
            if visited[i] is False:
                cc = self.__dfsHelper(i,visited,[])
                finalList.append(cc)  
        return finalList
    
g = Graph(8)
g.addEdge(0,7)
g.addEdge(1,2)
g.addEdge(1,4)
g.addEdge(3,5)
g.addEdge(3,6)
conn = g.connectedComponent()
print(conn)
    


# Minimum Spanning Tree (Kruskal Algorithm)

# In[4]:


class Edge:
    def __init__(self, src,dest,wt):
        self.src = src
        self.dest = dest
        self.wt = wt
        

def getParent(v,parent):
    if v==parent[v]:
        return v
    return getParent(parent[v],parent)
        
def kruskal(edges,nVertices):
    parent = [i for i in range(nVertices)]
    edges = sorted(edges,key = lambda edge:edge.wt)
    count = 0
    output = []
    i=0
    while count<(nVertices - 1):
        currentEdge = edges[i]
        srcParent =  getParent(currentEdge.src,parent)
        destParent = getParent(currentEdge.dest,parent)
        
        if srcParent!=destParent:
            output.append(currentEdge)
            count+=1
            parent[srcParent] = destParent
        i+=1
    return output   
            
li = [int(x) for x in input().split()]  
n = li[0]
E = li[1]
edges = []     
     
for i in range(E):
    curr_input = [int(c) for c in input().split()]
    src = curr_input[0]
    dest = curr_input[1]
    wt = curr_input[2]
    edge = Edge(src,dest,wt)
    edges.append(edge)

output = kruskal(edges,n)
for edge in output:
    if edge.src<edge.dest:
        print(str(edge.src) + " " + str(edge.dest) + " " + str(edge.wt) )
    else:
        print(str(edge.dest) + " " + str(edge.src) + " " + str(edge.wt) )
        
    


# In[4]:


print(ord('A'))


# In[31]:


s = 'opqrs'
l=[]
for i in s:
    l.append(ord(i))
l2=[]    
for i in range(1,len(l)):
    l2.append(abs(l[i]-l[i-1]))
for


# In[29]:


import numpy as np
ini_list = [111,112,113,114,115] 
print("intial_list", (ini_list)) 

# Calculating difference list 
diff_list = [] 

diff_list = np.diff(ini_list) 

# printing difference list 
print ("difference list: ", (diff_list)) 


# In[32]:


l =[ 1,2,3]
l.reverse()
l


# In[1]:


pr = -1
print(type(pr))


# In[ ]:




